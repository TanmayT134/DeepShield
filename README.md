1. Download the Celeb-DF dataset.
2. Place it in:

dataset/raw/

3. Run:

python -m src.preprocessing.process_dataset

4. Run:

python -m src.preprocessing.create_cnn_dataset

5. Train:

python main.py

6. Test:

python -m tests.test_video_predictor