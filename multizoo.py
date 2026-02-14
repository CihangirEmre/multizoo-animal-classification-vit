import customtkinter as ctk
from customtkinter import CTkImage
from tkinter import filedialog
from PIL import Image
import torch
import torchvision.transforms as transforms
import timm
import torch.nn as nn
import os

# === Customtkinter tema ayarı ===
ctk.set_appearance_mode("Dark")  # "Light" da olabilir
ctk.set_default_color_theme("blue")  # Tema: blue, green, dark-blue

# === Model Yükleme ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f]

num_classes = len(class_names)
model = timm.create_model('vit_base_patch16_224', pretrained=False)
model.head = nn.Linear(model.head.in_features, num_classes)
model.load_state_dict(torch.load("best_vit_model.pt", map_location=device))
model.to(device)
model.eval()
#Vit kendi yapıyor Data Augmentation
# === Görsel Dönüştürücü ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),#Boyutlandırma
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)#Normalizasyon
])

# === Tahmin Fonksiyonu ===
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        top3_probs, top3_indices = torch.topk(probabilities, 3)

        top3_results = []
        for i in range(3):
            label = class_names[top3_indices[i].item()]
            score = top3_probs[i].item()
            top3_results.append((label, score))

    return top3_results  # Liste: [(label1, score1), (label2, score2), (label3, score3)]

# === Arayüz Uygulaması ===
class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("MultiZoo - Hayvan Tahmini")
        self.geometry("500x600")
        self.center_window(500, 600)

        self.image_label = ctk.CTkLabel(self, text="Görsel Yüklenmedi")
        self.image_label.pack(pady=20)

        self.select_button = ctk.CTkButton(self, text="📂 Görsel Seç", command=self.load_image)
        self.select_button.pack(pady=10)

        self.result_label = ctk.CTkLabel(self, text="", font=("Arial", 16))
        self.result_label.pack(pady=10)

    def center_window(self, width=500, height=600):
        self.update_idletasks()
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Resim Dosyaları", "*.jpg *.png *.jpeg")])
        if file_path:
            self.current_image_path = file_path  

            image = Image.open(file_path).resize((224, 224))
            img = CTkImage(light_image=image, size=(224, 224))
            self.image_label.configure(image=img, text="")
            self.image_label.image = img

            self.image_label.bind("<Button-1>", self.open_fullscreen_image)  # 🖱 tıklanabilir yap

            top3 = predict_image(file_path)
            result_text = ""
            for i, (label, score) in enumerate(top3, 1):
                result_text += f"{i}. {label} - {score*100:.2f}%\n"

            self.result_label.configure(text=result_text.strip())

    def open_fullscreen_image(self, event=None):
        if not hasattr(self, "current_image_path") or not os.path.exists(self.current_image_path):
            return

        preview = ctk.CTkToplevel(self)
        preview.title("Görsel Önizleme")
        preview.geometry("800x800")
        preview.grab_set()

        image = Image.open(self.current_image_path).resize((700, 700))
        img = CTkImage(light_image=image, size=(700, 700))

        label = ctk.CTkLabel(preview, image=img, text="")
        label.image = img  
        label.pack(padx=20, pady=20)

if __name__ == "__main__":
    app = App()
    app.mainloop()
