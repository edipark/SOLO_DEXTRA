import json
import os
import argparse

def json_to_latex(file_path, output_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from '{file_path}'.")
        return

    config = data.get("config", {})
    results = data.get("results", {})

    # 1. 최적 성능 탐색 (Bold 처리를 위함)
    model_only = {k: v for k, v in results.items() if k.lower() != "baseline"}
    if model_only:
        best_len = max(v.get("avg_episode_mean", 0) for v in model_only.values())
        best_death = min(v.get("death_rate_mean", 100) for v in model_only.values())
    else:
        best_len, best_death = None, None

    # 2. LaTeX 서두 및 패키지 설정
    latex = [
        "\\documentclass{article}",
        "",
        "% --- Essential Packages ---",
        "\\usepackage[utf8]{inputenc}",
        "\\usepackage{booktabs} % For high-quality tables",
        "\\usepackage{geometry} % For page margins",
        "\\usepackage{caption}  % For caption styling",
        "\\usepackage{amsmath}  % For mathematical symbols",
        "\\usepackage{graphicx} % For \\resizebox",
        "\\geometry{a4paper, margin=1in}",
        "",
        "\\title{SOLO Ablation}",
        "",
        "\\begin{document}",
        "",
        "\\maketitle",
        "",
        "\\section{Experimental Configuration}",
        "\\begin{table}[ht]",
        "\\centering",
        "\\caption{Hyperparameter Settings and Environment Configuration}",
        "\\label{tab:config}",
        "\\begin{tabular}{@{}lc|lc@{}}",
        "\\toprule",
        "\\textbf{Configuration} & \\textbf{Value} & \\textbf{Configuration} & \\textbf{Value} \\\\ \\midrule"
    ]

    # Config 파싱 (2열 배치 로직 - 백슬래시 에러 방지 버전)
    config_items = list(config.items())
    for i in range(0, len(config_items), 2):
        pair = config_items[i:i+2]
        
        # 첫 번째 항목 처리
        k1, v1 = pair[0]
        clean_k1 = k1.replace('_', ' ').title()
        clean_v1 = str(v1).replace('_', '\\_')
        
        if len(pair) > 1:
            # 두 번째 항목 처리
            k2, v2 = pair[1]
            clean_k2 = k2.replace('_', ' ').title()
            clean_v2 = str(v2).replace('_', '\\_')
            row_str = f"{clean_k1} & {clean_v1} & {clean_k2} & {clean_v2} \\\\"
        else:
            row_str = f"{clean_k1} & {clean_v1} & & \\\\"
        
        latex.append(row_str)

    latex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
        "\\section{Performance Evaluation}",
        "",
        "\\begin{table}[ht]",
        "\\centering",
        "\\caption{Performance Comparison across Architectures and Window Sizes}",
        "\\label{tab:results}",
        "\\resizebox{\\textwidth}{!}{",
        "\\begin{tabular}{@{}lccccc@{}}",
        "\\toprule",
        "\\textbf{Model Index} & \\textbf{Window} & \\textbf{DAgger} & \\textbf{Avg. Episode Length} & \\textbf{Death Rate (\\%)} & \\textbf{Status} \\\\ \\midrule"
    ])

    # 3. Results 파싱 (Baseline 우선 정렬)
    sorted_keys = sorted(results.keys(), key=lambda x: (x.lower() != "baseline", x))
    
    for key in sorted_keys:
        val = results[key]
        safe_key = key.replace("_", "\\_")
        is_baseline = (key.lower() == "baseline")
        
        window = val.get("window", "--")
        dagger = "Yes" if val.get("use_dagger", False) else ("--" if is_baseline else "No")
        
        mean_len = val.get("avg_episode_mean", 0)
        std_len = val.get("avg_episode_std", 0)
        mean_death = val.get("death_rate_mean", 0)
        std_death = val.get("death_rate_std", 0)
        
        len_str = f"{mean_len:.2f} \\pm {std_len:.2f}"
        death_str = f"{mean_death:.2f} \\pm {std_death:.2f}"
        
        # Status 판별
        if is_baseline:
            status = "Optimal"
            key_display = f"\\textit{{{safe_key}}}"
        else:
            if mean_death == 0: status = "Success"
            elif mean_death < 15: status = "Marginal"
            elif mean_death < 50: status = "Unstable"
            else: status = "Failed"
            
            # 최적값 볼드체 적용
            if mean_len == best_len:
                len_str = f"\\mathbf{{{len_str}}}"
            if mean_death == best_death:
                death_str = f"\\mathbf{{{death_str}}}"
            
            key_display = safe_key

        latex.append(f"{key_display} & {window} & {dagger} & ${len_str}$ & ${death_str}$ & {status} \\\\")

    latex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\end{table}",
        "",
        "\\end{document}"
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument("-o", "--output", default="report.tex")
    args = parser.parse_args()

    json_to_latex(args.json, args.output)
    print(f"✅ LaTeX report generated: {args.output}")

if __name__ == "__main__":
    main()