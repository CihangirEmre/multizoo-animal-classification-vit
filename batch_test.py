import os
from PIL import Image
import torch
import torchvision.transforms as transforms
import timm
import torch.nn as nn

# === MODEL ve SINIFLAR ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f]

num_classes = len(class_names)
model = timm.create_model('vit_base_patch16_224', pretrained=False)
model.head = nn.Linear(model.head.in_features, num_classes)
model.load_state_dict(torch.load("best_vit_model.pt", map_location=device))
model.to(device)
model.eval()

# === DÖNÜŞÜMLER ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# === Tahmin fonksiyonu ===
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top3_probs, top3_indices = torch.topk(probabilities, 3)

        results = []
        for i in range(3):
            label = class_names[top3_indices[i].item()]
            score = top3_probs[i].item()
            results.append((label, score))
    return results

# === KLASÖRDEKİ TÜM GÖRSELLERİ İŞLE ===
test_dir = "test/test"  
output_file = "test_sonuc.txt"

#Bütün klasörün çıktısını verir
count = 0

with open(output_file, "w", encoding="utf-8") as out:
    for root, _, files in os.walk(test_dir):
        for filename in files:
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(root, filename)
                try:
                    predictions = predict_image(image_path)
                except Exception as e:
                    print(f"Error on {image_path}: {e}")
                    continue

                out.write(f"{filename} ({os.path.basename(root)}):\n")
                for i, (label, score) in enumerate(predictions, 1):
                    out.write(f"  {i}. {label} - {score*100:.2f}%\n")
                out.write("\n")
                count += 1

#Sadece Yanlış tahmin ve Düşük Güven sonuçlarını verir
# with open(output_file, "w", encoding="utf-8") as out:
#     for root, _, files in os.walk(test_dir):
#         true_label = os.path.basename(root)  

#         for filename in files:
#             if filename.lower().endswith((".jpg", ".jpeg", ".png")):
#                 image_path = os.path.join(root, filename)
#                 try:
#                     predictions = predict_image(image_path)
#                 except Exception as e:
#                     print(f"Error: {e} - {image_path}")
#                     continue

#                 predicted_label, predicted_score = predictions[0]

                
#                 if predicted_label == true_label and predicted_score < 0.60:
#                     out.write(f"[LOW CONFIDENCE] {filename} (True: {true_label})\n")
#                     out.write(f"  1. {predicted_label} - {predicted_score*100:.2f}%\n")
#                     out.write("\n")

                
#                 elif predicted_label != true_label:
#                     out.write(f"[WRONG PREDICTION] {filename} (True: {true_label})\n")
#                     for i, (label, score) in enumerate(predictions, 1):
#                         out.write(f"  {i}. {label} - {score*100:.2f}%\n")
#                     out.write("\n")



print(f"✅ Tahminler '{output_file}' dosyasına yazıldı.")

#Metrik hesaplama
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Gerçek ve tahmin edilen etiketleri tut
y_true = []
y_pred = []

for root, _, files in os.walk(test_dir):
    true_label = os.path.basename(root)

    for filename in files:
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(root, filename)
            try:
                predictions = predict_image(image_path)
            except Exception as e:
                print(f"Error: {e} - {image_path}")
                continue

            predicted_label = predictions[0][0]  # En yüksek olasılıklı tahmin

            y_true.append(true_label)
            y_pred.append(predicted_label)

# Skorları hesapla
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="macro")
recall = recall_score(y_true, y_pred, average="macro")
f1 = f1_score(y_true, y_pred, average="macro")

print("\n📊 Değerlendirme Metrikleri:")
print(f"✅ Accuracy: {accuracy * 100:.2f}%")
print(f"✅ Precision (macro): {precision * 100:.2f}%")
print(f"✅ Recall (macro): {recall * 100:.2f}%")
print(f"✅ F1 Score (macro): {f1 * 100:.2f}%")

import csv

with open("test_metrics.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Metric", "Value (%)"])
    writer.writerow(["Accuracy", f"{accuracy * 100:.2f}"])
    writer.writerow(["Precision", f"{precision * 100:.2f}"])
    writer.writerow(["Recall", f"{recall * 100:.2f}"])
    writer.writerow(["F1 Score", f"{f1 * 100:.2f}"])

