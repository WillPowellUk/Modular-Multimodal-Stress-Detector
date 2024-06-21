import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, confusion_matrix, log_loss, 
                             precision_score, recall_score, f1_score, ConfusionMatrixDisplay)

class ModelResultsAnalysis:
    def __init__(self, results):
        self.results = results

    def plot_confusion_matrix(self, num_of_labels, cm=0, y_test=None, y_pred=None):
        display_labels = ['Low', 'High'] if num_of_labels == 2 else ['Low', 'Medium', 'High']
        default_font_size = plt.rcParams['font.size']
        plt.rcParams.update({'font.size': default_font_size * 1.4})
        
        if isinstance(cm, int):
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=display_labels, normalize='true')
        else:
            disp = ConfusionMatrixDisplay(cm, display_labels=display_labels)
            disp.plot(values_format='.2f')
            plt.show()
        
        plt.rcParams.update({'font.size': default_font_size})

    def print_metrics(self, metrics):
        for model, metric in metrics.items():
            print(f"{model}:")
            print(f"  Accuracy: {metric['accuracy']:.5f}")
            if metric['loss'] is not None:
                print(f"  Loss: {metric['loss']:.5f}")
            print(f"  Precision: {metric['precision']:.5f}")
            print(f"  Recall: {metric['recall']:.5f}")
            print(f"  F1 Score: {metric['f1_score']:.5f}")

    def analyze_subject(self, subject_id):
        if subject_id < 0 or subject_id >= len(self.results):
            raise ValueError("Invalid subject ID")
        
        subject_results = self.results[subject_id]
        self.print_metrics(subject_results)
        
        for model_name, metrics in subject_results.items():
            num_of_labels = len(metrics['confusion_matrix'])
            self.plot_confusion_matrix(num_of_labels, cm=metrics['confusion_matrix'])

    def analyze_collective(self):
        collective_metrics = {}
        for model_name in self.results[0].keys():
            accuracies = []
            losses = []
            precisions = []
            recalls = []
            f1_scores = []
            for subject_results in self.results:
                accuracies.append(subject_results[model_name]['accuracy'])
                if subject_results[model_name]['loss'] is not None:
                    losses.append(subject_results[model_name]['loss'])
                precisions.append(subject_results[model_name]['precision'])
                recalls.append(subject_results[model_name]['recall'])
                f1_scores.append(subject_results[model_name]['f1_score'])
            
            avg_accuracy = np.mean(accuracies)
            avg_loss = np.mean(losses) if losses else None
            avg_precision = np.mean(precisions)
            avg_recall = np.mean(recalls)
            avg_f1_score = np.mean(f1_scores)
            
            collective_metrics[model_name] = {
                'accuracy': avg_accuracy,
                'loss': avg_loss,
                'precision': avg_precision,
                'recall': avg_recall,
                'f1_score': avg_f1_score
            }

        self.print_metrics(collective_metrics)

        for model_name, metrics in collective_metrics.items():
            all_cm = np.sum([subject_results[model_name]['confusion_matrix'] for subject_results in self.results], axis=0)
            num_of_labels = len(all_cm)
            self.plot_confusion_matrix(num_of_labels, cm=all_cm)

    def plot_confusion_matrix(self, num_labels, cm):
        plt.figure(figsize=(10, 7))
        ax = plt.gca()
        
        cm_sum = np.sum(cm, axis=1, keepdims=True)
        cm_percentage = cm / cm_sum.astype(float) * 100

        labels = np.asarray([f"{value}\n{percentage:.2f}%" for value, percentage in zip(cm.flatten(), cm_percentage.flatten())]).reshape(cm.shape)
        # labels = np.asarray([f"{percentage:.2f}%" for value, percentage in zip(cm.flatten(), cm_percentage.flatten())]).reshape(cm.shape)
        sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=True, ax=ax, annot_kws={"size": 14})

        ax.set_xlabel('Predicted labels', fontsize=16)
        ax.set_ylabel('True labels', fontsize=16)

        if num_labels == 2:
            ax.set_xticklabels(['Non-Stressed', 'Stressed'], fontsize=14)
            ax.set_yticklabels(['Non-Stressed', 'Stressed'], fontsize=14)
        else:
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)

        plt.show()
