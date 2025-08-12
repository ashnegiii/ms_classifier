from utils import build_train, build_test

train_prefixes = ["02-01-01", "03-04-03"]
test_prefixes  = ["02-04-04"]

build_train(prefixes=train_prefixes,
            video_dir="data/videos",
            csv_dir="data/train_labels",
            data_root="data",
            frame_percentage=1)

build_test(prefixes=test_prefixes,
           video_dir="data/videos",
           csv_dir="data/test_labels",
           data_root="data",
           frame_percentage=1)
