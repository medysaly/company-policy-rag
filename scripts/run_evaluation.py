from app.evaluation.evaluate import run_evaluation

if __name__ == "__main__":
    run_evaluation(
        handbook_path="data/sample/company_handbook.txt",
        eval_dataset_path="data/eval_dataset.json",
    )

