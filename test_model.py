import os
from waterlogging_detection import predict_waterlogging

test_dir = 'analyis'
if os.path.exists(test_dir):
    print("Testing model with analysis images:")
    print("-" * 60)
    for img in sorted(os.listdir(test_dir)):
        if img.endswith('.jpg'):
            path = os.path.join(test_dir, img)
            label, conf, prob = predict_waterlogging(path)
            print(f"{img:15} -> {label:20} Confidence: {conf*100:6.2f}%")
else:
    print("analyis folder not found")
