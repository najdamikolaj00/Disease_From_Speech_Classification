import re
from itertools import chain
from pathlib import Path

from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    rek_data_path = Path('Data/Vowels/Rekurrensparese')
    all_patients_rek = tuple(set(chain.from_iterable(map(re.compile(r"\d+").findall, map(str, rek_data_path.iterdir())))))
    train_patients_rek, test_patients_rek = train_test_split(all_patients_rek, test_size=.2, random_state=42)
    train_recordings_rek = tuple(filter(lambda path: any(map(re.findall(r"\d+", path.name)[0].__eq__, train_patients_rek)), rek_data_path.iterdir()))
    test_recordings_rek = tuple(filter(lambda path: any(map(re.findall(r"\d+", path.name)[0].__eq__, test_patients_rek)), rek_data_path.iterdir()))
    assert not tuple(filter(test_recordings_rek.__contains__, train_recordings_rek))
    data_path = Path('Data/Vowels/Healthy')
    all_patients = tuple(set(chain.from_iterable(map(re.compile(r"\d+").findall, map(str, data_path.iterdir())))))
    train_patients, test_patients = train_test_split(all_patients, test_size=.2, random_state=42)
    train_recordings = tuple(filter(lambda path: any(map(re.findall(r"\d+", path.name)[0].__eq__, train_patients)), data_path.iterdir()))
    test_recordings = tuple(filter(lambda path: any(map(re.findall(r"\d+", path.name)[0].__eq__, test_patients)), data_path.iterdir()))
    assert not tuple(filter(test_recordings.__contains__, train_recordings))
    Path('Data/Lists/Vowels_all_Rekurrensparese_train.txt').write_text('\n'.join(f"Vowels/Rekurrensparese/{path.name} 1" for path in train_recordings_rek) + '\n' + '\n'.join(f"Vowels/Healthy/{path.name} 0" for path in train_recordings))
    Path('Data/Lists/Vowels_all_Rekurrensparese_test.txt').write_text('\n'.join(f"Vowels/Rekurrensparese/{path.name} 1" for path in test_recordings_rek) + '\n' + '\n'.join(f"Vowels/Healthy/{path.name} 0" for path in test_recordings))

