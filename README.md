# CLIP training
Обучение модели CLIP на основе данных, предоставленных Яндексом в рамках соревнования [ML challenge Yandex Cup 2021 (CV)](https://yandex.ru/cup/ml/analysis/#CV)
## Настройка конфига
`config.yaml` - файл с конфигом. Необходимо указать:
- `pro_path` - путь до папки с проектом
- `image_urls_path` - путь до [images.json](https://cvlab.s3.yandex.net/mlcup2021/images.json) - файл с ссылками на изображения
- `captions_path` - путь до [captions.json](https://cvlab.s3.yandex.net/mlcup2021/metadata.json) - файл с подписями к изображениям
- `images_path` - папка, в которую будут сохраняться скачанные изображения и использоваться для обучения модели
- `val_data_path` - путь к папке с [валидационной выборке](https://github.com/yandex/mlcup/tree/main/cv/contest/data/)
- `val_gt_path` - путь к файлу с ground truth метками валидационных данных

## Скрипты
- `download.py` - скрипт для скачивания данных
- `train.py` - скрипт для начала обучения модели
