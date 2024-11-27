from django.shortcuts import render
from .forms import ImageUploadForm
from django.conf import settings
from tensorflow.keras.models import load_model 
from tensorflow.keras.applications.vgg16 import decode_predictions, preprocess_input 
from tensorflow.keras.preprocessing.image import load_img, img_to_array 
from io import BytesIO
import os

# モデルのロード
model_path = os.path.join(settings.BASE_DIR, 'prediction', 'models', 'vgg16.h5')
vgg_model = load_model(model_path)

def predict(request):
    if request.method == 'GET':
        form = ImageUploadForm()
        return render(request, 'home.html', {'form': form})

    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            img_file = form.cleaned_data['image']

            # 画像の前処理
            img_file = BytesIO(img_file.read())
            img = load_img(img_file, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = img_array.reshape((1, 224, 224, 3))
            img_array = preprocess_input(img_array)  # VGG16の前処理

            # 推論
            predictions = vgg_model.predict(img_array)
            decoded_predictions = decode_predictions(predictions, top=5)
            
            result = [
                {'category': pred[1], 'probability': round(pred[2] * 100, 2)}
                for pred in decoded_predictions[0]
            ]

            img_data = request.POST.get('img_data')
            return render(request, 'home.html', {'form': form, 'result':result, 'img_data': img_data})
        else:
            form = ImageUploadForm()
            return render(request, 'home.html', {'form': form})

