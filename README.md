
# 🌿 Leaf_AI

Leaf_AI is a deep learning–powered web application that classifies leaf images into their corresponding plant species. Built using **TensorFlow** and **Streamlit**, the model supports over 35 species — identifying whether a leaf belongs to **Angiospermae** or **Gymnospermae** plant groups.

## 🚀 Features

- 🌱 Classifies leaf images into detailed Latin plant species names
- 🧬 Differentiates between Gymnospermae and Angiospermae
- 📸 Supports both image upload and real-time camera input
- 📊 Provides confidence score with each prediction
- 💻 Streamlit web interface for ease of use

## 🧠 Model Details

- Image input size: `150x150`
- Model format: `Keras (.keras)`
- Custom evaluation metric: F1-score
- Prediction pipeline includes preprocessing with OpenCV and Keras

## 🛠 Tech Stack

- **Python**
- **TensorFlow / Keras**
- **OpenCV**
- **Streamlit**
- **NumPy / PIL**

## 📦 Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/HadiR-13/Leaf_AI.git
   cd Leaf_AI
   ```

2. **Create virtual environment (optional but recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure your model file is available**

   * Model file: `leaf_latin_model.keras`
   * Test images folder: `./Test_AI/`

## ▶️ Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

Use the sidebar to choose between uploading an image or using your device's camera.


## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/foo`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/foo`)
5. Create a new Pull Request

## 📜 License

This project is licensed under the MIT License.

