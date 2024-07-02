import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import sys
from IPython.display import display, Latex
import time


class ModelResultsAnalysis:
    def __init__(self, results):
        self.results = results

    def print_metrics(self, metrics):
        for model, metric in metrics.items():
            print(f"{model}:")
            if "subject_id" in metric:
                print(f"  Subject ID: {metric['subject_id']}")
            print(f"  Accuracy: {metric['accuracy']:.5f}")
            print(f"  Precision: {metric['precision']:.5f}")
            print(f"  Recall: {metric['recall']:.5f}")
            print(f"  F1 Score: {metric['f1_score']:.5f}")
            print(f"  Inference Time (ms): {metric['inference_time_ms']:.5f}")

    def analyze_subject(self, subject_id):
        if subject_id < 0 or subject_id >= len(self.results):
            raise ValueError("Invalid subject ID")

        subject_results = self.results[subject_id]
        self.print_metrics(subject_results)

        for model_name, metrics in subject_results.items():
            num_of_labels = len(metrics["confusion_matrix"])
            self.plot_confusion_matrix(num_of_labels, cm=metrics["confusion_matrix"])

    def plot_confusion_matrix(self, num_labels, cm):
        fig = plt.figure(figsize=(10, 7))
        ax = plt.gca()

        cm_sum = np.sum(cm, axis=1, keepdims=True)
        cm_percentage = cm / cm_sum.astype(float) * 100

        labels = np.asarray(
            [
                f"{value}\n{percentage:.2f}%"
                for value, percentage in zip(cm.flatten(), cm_percentage.flatten())
            ]
        ).reshape(cm.shape)
        # labels = np.asarray([f"{percentage:.2f}%" for value, percentage in zip(cm.flatten(), cm_percentage.flatten())]).reshape(cm.shape)
        sns.heatmap(
            cm,
            annot=labels,
            fmt="",
            cmap="Blues",
            cbar=True,
            ax=ax,
            annot_kws={"size": 18},
        )

        ax.set_xlabel("Predicted labels", fontsize=20)
        ax.set_ylabel("True labels", fontsize=20)

        if num_labels == 2:
            ax.set_xticklabels(["Non-Stressed", "Stressed"], fontsize=18)
            ax.set_yticklabels(["Non-Stressed", "Stressed"], fontsize=18)
        else:
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)

        # plt.show()
        return fig

    def analyze_collective(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        collective_metrics = {}
        all_results = []

        for model_name in self.results[0].keys():
            accuracies = []
            precisions = []
            recalls = []
            f1_scores = []
            inference_times = []
            model_results = []

            for subject_results in self.results:
                subject_id = subject_results[model_name]["subject_id"]
                accuracies.append(subject_results[model_name]["accuracy"])
                precisions.append(subject_results[model_name]["precision"])
                recalls.append(subject_results[model_name]["recall"])
                f1_scores.append(subject_results[model_name]["f1_score"])
                inference_times.append(subject_results[model_name]["inference_time_ms"])

                result_entry = {
                    "subject_id": subject_id,
                    "model_name": model_name,
                    "accuracy": subject_results[model_name]["accuracy"],
                    "precision": subject_results[model_name]["precision"],
                    "recall": subject_results[model_name]["recall"],
                    "f1_score": subject_results[model_name]["f1_score"],
                    "inference_time_ms": subject_results[model_name][
                        "inference_time_ms"
                    ],
                }

                all_results.append(result_entry)
                model_results.append(result_entry)

            avg_accuracy = np.mean(accuracies)
            avg_precision = np.mean(precisions)
            avg_recall = np.mean(recalls)
            avg_f1_score = np.mean(f1_scores)
            avg_inference_time = np.mean(inference_times)

            collective_metrics[model_name] = {
                "accuracy": avg_accuracy,
                "precision": avg_precision,
                "recall": avg_recall,
                "f1_score": avg_f1_score,
                "inference_time_ms": avg_inference_time,
            }

            # Convert model-specific results to DataFrame
            model_results_df = pd.DataFrame(model_results)
            display(model_results_df)

            # Convert DataFrame to LaTeX and save to file
            latex_table = model_results_df.to_latex(index=False)
            latex_file_path = os.path.join(save_dir, f"{model_name}_results_table.tex")
            with open(latex_file_path, "w") as f:
                f.write(latex_table)
            print(f"LaTeX Table for {model_name} saved to {latex_file_path}\n")

        print(f"Collective metrics:")
        self.print_metrics(collective_metrics)

        for model_name, metrics in collective_metrics.items():
            all_cm = np.sum(
                [
                    subject_results[model_name]["confusion_matrix"]
                    for subject_results in self.results
                ],
                axis=0,
            )
            num_of_labels = len(all_cm)
            fig = self.plot_confusion_matrix(num_of_labels, cm=all_cm)
            confusion_matrix_path = os.path.join(
                save_dir, f"{model_name}_confusion_matrix.png"
            )
            fig.savefig(
                confusion_matrix_path, dpi=300, format="png", bbox_inches="tight"
            )
            plt.close(fig)
            print(f"Confusion matrix for {model_name} saved to {confusion_matrix_path}")

        # Plot bar chart for each model's validation accuracy for each subject ID
        for model_name in pd.DataFrame(all_results)["model_name"].unique():
            model_data = pd.DataFrame(all_results)[
                pd.DataFrame(all_results)["model_name"] == model_name
            ]
            plt.figure(figsize=(10, 6))
            bar1 = plt.bar(
                model_data["subject_id"], model_data["accuracy"], color="darkslategray"
            )
            for i, bar in enumerate(bar1):
                yval = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    yval,
                    round(yval, 2),
                    va="bottom",
                    ha="center",
                    fontsize=18,
                )
            plt.xlabel("Subject ID", fontsize=24)
            plt.ylabel("Validation Accuracy", fontsize=24)
            plt.ylim((0, 1))  # Adjusting the y-limit to accommodate runtime text
            unique_subject_ids = model_data["subject_id"].unique()
            plt.xticks(
                unique_subject_ids, fontsize=18
            )  # Setting x-ticks to be the unique subject IDs
            plt.yticks(fontsize=18)
            bar_chart_path = os.path.join(
                save_dir, f"{model_name}_validation_accuracy.png"
            )
            plt.savefig(bar_chart_path, dpi=300, format="png", bbox_inches="tight")
            # plt.show()
            print(
                f"Validation accuracy bar chart for {model_name} saved to {bar_chart_path}"
            )
