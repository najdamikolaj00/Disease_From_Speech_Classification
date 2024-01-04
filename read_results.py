import re
from pathlib import Path

if __name__ == '__main__':
    # for model_name, accuracy, f1, precision, recall in re.findall(
    #     r'(\w+)\s*Test\s*Accuracy:\s*([\d\.]+)%\s*F1-score:\s*([\d\.]+),\s*Precision:\s*([\d\.]+),\s*Recall:\s*([\d\.]+)',
    #     Path('results.txt').read_text()):
    ' \\\ \hline\n'.join(map(' & '.join, sorted(re.findall(
        r'(\w+)\s*Test\s*Accuracy:\s*([\d\.]+%)\s*F1-score:\s*([\d\.]+),\s*Precision:\s*([\d\.]+),\s*Recall:\s*([\d\.]+)',
        Path('results.txt').read_text()), key=lambda item: float(item[2]), reverse=True)))
