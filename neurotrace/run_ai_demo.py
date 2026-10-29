"""Quick check: python run_ai_demo.py [image_path]"""
import json
import sys
from pathlib import Path

import config
from ai.predict import predict
from ai.explain import explain

img = Path(sys.argv[1]) if len(sys.argv) > 1 else config.DEFAULT_IMAGE
pred = predict(img)
print(json.dumps(pred, indent=2))
print(json.dumps(explain(img, pred, case_id="CASE_DEMO"), indent=2))

# ---- preprocessing + prediction chain ----
from preprocessing.preprocess import preprocess_image
res, model_img = preprocess_image(img, "CASE_DEMO")
print(json.dumps(res, indent=2))
if res["valid"]:
    print(json.dumps(predict(model_img), indent=2))

# ---- validation ----
from validation.validate import validate
if res["valid"]:
    print(json.dumps(validate(res, predict(model_img)), indent=2, ensure_ascii=False))