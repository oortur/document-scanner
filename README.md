# Document scanner + target word recognizer
Document scanner: input photo of your document to get its smooth pdf version. Optionally recognize words in your document and mark them.

## Getting started

### Scanner

At first, clone the repo:
```bash
git clone https://github.com/balan/document-scanner.git
```

I assume you have `numpy` installed in your python. Next you need to install computer vision tools:
```bash
pip install opencv-python scikit-image
```
  
This is all you need to run your document scanner. 

### Word recognizer

If word recognizer option is also needed, install Tesseract OCR engine:
```bash
sudo apt install tesseract-ocr libtesseract-dev
pip install pytesseract
```
  
The default language of Tesseract is English, if your document is written in other language install special package at first (lang responds to your language):
```bash
sudo apt-get install tesseract-ocr-[lang]
```

## Usage

Simply run `python scan.py` with proper arguments to get result. Observe the application interface:
```bash
usage: document-scanner [-h] [-i IMAGE] [-p PDF] [--binarize] [-w [TARGET_WORDS [TARGET_WORDS ...]]] [-l LANG]

optional arguments:
  -h, --help            show this help message and exit
  -i IMAGE, --image IMAGE
                        path to input image
  -p PDF, --pdf PDF     path to store document in pdf format
  --binarize            save two document versions: original and binarized (stores both by default, only original when false)
  -w [TARGET_WORDS [TARGET_WORDS ...]], --words [TARGET_WORDS [TARGET_WORDS ...]]
                        list of target words to search in document (no words by default)
  -l LANG, --lang LANG  specify language of text if needed (english by default)
```

## Examples

You may find examples of input images and pdf files in `/images` and `/out` folders respectively.

Input and otuput with recognized target words:

<img src="https://github.com/balan/document-scanner/blob/main/images/sample_0.jpg?raw=true" height=300 hspace=30> <img src="https://github.com/balan/document-scanner/blob/main/out/out_box.jpg?raw=true" height=300>

Observe marked words connected to <b>sign</b> and <b>signature</b>:

<img src="https://github.com/balan/document-scanner/blob/main/out/out_box_close.jpg?raw=true" height=200>
