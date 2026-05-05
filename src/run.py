import json
import subprocess
import sys
import time
from pathlib import Path
from src.data import get_samples
from src.react import run_react

sys.stdout.reconfigure(encoding="utf-8")


def _notify(title: str, msg: str):
    if sys.platform != "win32":
        return
    try:
        ps = f"""
Add-Type -AssemblyName System.Windows.Forms
$n = New-Object System.Windows.Forms.NotifyIcon
$n.Icon = [System.Drawing.SystemIcons]::Information
$n.BalloonTipTitle = "{title}"
$n.BalloonTipText = "{msg}"
$n.Visible = $True
$n.ShowBalloonTip(8000)
Start-Sleep -Milliseconds 8500
$n.Dispose()
"""
        subprocess.Popen(
            ["powershell", "-WindowStyle", "Hidden", "-Command", ps],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass

VARIANTS = ["full", "none", "compressed"]
LOG_DIR = Path("logs")


def _normalize(text: str) -> str:
    import re
    text = text.strip().lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def _extract_number(text: str):
    import re
    # extract last number (int or float) from text
    nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text.replace(",", ""))
    if nums:
        try:
            return float(nums[-1])
        except ValueError:
            pass
    return None


def evaluate(pred: str, aliases: list[str]) -> dict:
    pred_norm = _normalize(pred) if pred else ""
    # numeric comparison: if gold looks like a number, extract and compare numerically
    gold_num = _extract_number(aliases[0]) if aliases else None
    if gold_num is not None:
        pred_num = _extract_number(pred) if pred else None
        em = int(pred_num is not None and abs(pred_num - gold_num) < 1e-6)
        return {"em": em, "f1": float(em)}
    # fallback: token-level F1 against best alias
    em = int(any(pred_norm == _normalize(a) for a in aliases))
    best_f1 = 0.0
    pred_tokens = set(pred_norm.split())
    for alias in aliases:
        gold_tokens = set(_normalize(alias).split())
        if not gold_tokens:
            continue
        overlap = pred_tokens & gold_tokens
        precision = len(overlap) / len(pred_tokens) if pred_tokens else 0.0
        recall = len(overlap) / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        best_f1 = max(best_f1, f1)
    return {"em": em, "f1": round(best_f1, 4)}


def run_pilot(n: int = 10, model: str = "deepseek-chat", seed: int = 42, dataset: str = "hotpotqa", hotpotqa_type: str = None):
    LOG_DIR.mkdir(exist_ok=True)
    _notify("实验开始", f"正在加载数据集 {dataset}，n={n}...")
    samples = get_samples(n, seed=seed, dataset=dataset, hotpotqa_type=hotpotqa_type)
    _notify("数据加载完成", f"{dataset} {n} 条样本就绪，开始跑实验")

    for variant in VARIANTS:
        model_slug = model.replace("/", "_")
        task_tag = f"{dataset}_{hotpotqa_type}" if hotpotqa_type else dataset
        log_path = LOG_DIR / f"pilot_{variant}_{model_slug}_{task_tag}_n{n}_seed{seed}.jsonl"
        print(f"\n=== Variant: {variant} | Model: {model} | N={n} ===")

        # resume: load already-done IDs
        done_ids = set()
        results = []
        if log_path.exists():
            for line in log_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    done_ids.add(rec["id"])
                    results.append(rec)
            if done_ids:
                print(f"  Resuming: {len(done_ids)} already done, {n - len(done_ids)} remaining")

        try:
            with open(log_path, "a", encoding="utf-8") as f:
                for i, sample in enumerate(samples):
                    if sample["id"] in done_ids:
                        continue
                    print(f"  [{i+1}/{n}] {sample['question'][:60]}...")
                    t0 = time.time()
                    result = run_react(sample["question"], variant=variant, model=model)
                    elapsed = round(time.time() - t0, 2)

                    metrics = evaluate(result["answer"], sample["aliases"])
                    record = {
                        "id": sample["id"],
                        "question": sample["question"],
                        "gold": sample["answer"],
                        "pred": result["answer"],
                        "status": result["status"],
                        "steps": result["steps"],
                        "elapsed": elapsed,
                        **metrics,
                        "trajectory": result["trajectory"],
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    results.append(record)
                    print(f"    pred={str(result['answer'])[:40]!r}  gold={sample['answer']!r}  EM={metrics['em']}  F1={metrics['f1']}  steps={result['steps']}")
        except Exception as e:
            _notify("⚠️ 实验出错", f"{variant} 变体第 {len(results)} 条出错：{str(e)[:80]}")
            raise

        em_avg = round(sum(r["em"] for r in results) / len(results), 4)
        f1_avg = round(sum(r["f1"] for r in results) / len(results), 4)
        success = sum(1 for r in results if r["status"] == "success")
        print(f"  >> EM={em_avg}  F1={f1_avg}  success={success}/{n}  log={log_path}")
        _notify(f"✅ {variant} 完成", f"EM={em_avg}  F1={f1_avg}  success={success}/{n}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--model", type=str, default="deepseek-chat")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default="hotpotqa")
    parser.add_argument("--hotpotqa_type", type=str, default=None, help="bridge or comparison")
    args = parser.parse_args()
    run_pilot(n=args.n, model=args.model, seed=args.seed, dataset=args.dataset, hotpotqa_type=args.hotpotqa_type)
    _notify("🎉 全部完成", f"{args.dataset} n={args.n} 三个变体均已跑完")
