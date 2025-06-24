# ocr_utils.py
import pytesseract
from PIL import Image
import os

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi("mainwindow.ui", self)
        
        # Connect UI elements
        self.pushButton_browse.clicked.connect(self.browse_image)
        self.pushButton_extract.clicked.connect(self.extract_text)
        self.pushButton_confidence.clicked.connect(self.extract_with_confidence)
        
        self.image_path = ""
        
    def advanced_image_to_text(image_path, language='eng', config=''):
    """
    Advanced OCR with language and configuration options
    """
        try:
            if not os.path.exists(image_path):
                return f"Error: File '{image_path}' not found"
        
        # Open image
            img = Image.open(image_path)
        
        # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
        
        # Extract text with custom configuration
            text = pytesseract.image_to_string(
                img, 
                lang=language,
                config=config
            )
        
            return text.strip()
    
        except Exception as e:
            return f"Error: {e}"

    def get_text_with_confidence(image_path):
    """Get text with confidence scores"""
        try:
            img = Image.open(image_path)
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        
            text_with_confidence = []
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 0:
                    text_with_confidence.append({
                        'text': data['text'][i],
                        'confidence': data['conf'][i]
                    })
        
            return text_with_confidence
    
        except Exception as e:
            return f"Error: {e}"
            
    def browse_image(self):
        """Browse and select image file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Image", 
            "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        
        if file_path:
            self.image_path = file_path
            self.lineEdit_path.setText(file_path)
    
    def extract_text(self):
        """Extract text using enhanced OCR function"""
        if not self.image_path:
            QMessageBox.warning(self, "Warning", "Please select an image first!")
            return
        
        # Get configuration from UI elements
        language = self.get_selected_language()
        config = self.get_ocr_config()
        
        # Call the enhanced OCR function
        extracted_text = advanced_image_to_text(self.image_path, language, config)
        
        # Display result
        self.textEdit_result.setPlainText(extracted_text)
    
    def extract_with_confidence(self):
        """Extract text with confidence scores"""
        if not self.image_path:
            QMessageBox.warning(self, "Warning", "Please select an image first!")
            return
        
        confidence_data = get_text_with_confidence(self.image_path)
        
        # Format and display confidence data
        if isinstance(confidence_data, list):
            formatted_text = "\n".join([
                f"{item['text']} (confidence: {item['confidence']}%)" 
                for item in confidence_data if item['text'].strip()
            ])
            self.textEdit_result.setPlainText(formatted_text)
        else:
            self.textEdit_result.setPlainText(str(confidence_data))
    
    def get_selected_language(self):
        """Get language selection from UI"""
        if hasattr(self, 'comboBox_language'):
            lang_map = {
                'English': 'eng',
                'Spanish': 'spa',
                'French': 'fra',
                'German': 'deu'
            }
            return lang_map.get(self.comboBox_language.currentText(), 'eng')
        return 'eng'
    
    def get_ocr_config(self):
        """Get OCR configuration from UI checkboxes/options"""
        config = ""
        
        if hasattr(self, 'checkBox_numbers_only') and self.checkBox_numbers_only.isChecked():
            config += "--psm 6 -c tessedit_char_whitelist=0123456789"
        
        if hasattr(self, 'checkBox_single_line') and self.checkBox_single_line.isChecked():
            config += " --psm 7"
        
        return config.strip()