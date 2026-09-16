import sys
from pathlib import Path
from src.predict import ImagePredictor

img_path = Path("C:/Users/ASUS/.gemini/antigravity-ide/brain/6b22a9f7-eeaf-4b8e-8194-e3d56c491108/media__1786773624970.png")
if not img_path.exists():
    print(f"Image not found at {img_path}")
    sys.exit(1)

predictor = ImagePredictor("models/final_model.keras")
res = predictor.predict_image(str(img_path))

print("\n" + "=" * 50)
print(" PREDICTION RESULT FOR USER'S TEST IMAGE:")
print("=" * 50)
print(f" Predicted Class   : {res['prediction']}")
print(f" Confidence        : {res['confidence']*100:.2f}%")
print(f" Low Confidence    : {res['low_confidence']}")
print(" Class Probabilities:")
for k, v in res["probabilities"].items():
    print(f"   - {k:10s}: {v*100:.2f}%")
print("=" * 50)
