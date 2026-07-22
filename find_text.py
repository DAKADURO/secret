import pdfplumber, os
for f in os.listdir('datos/manuales/KAISHAN/KRSD-125'):
  try:
    with pdfplumber.open(f'datos/manuales/KAISHAN/KRSD-125/{f}') as pdf:
      for i, page in enumerate(pdf.pages):
        text = page.extract_text() or ''
        if '4000' in text or '4,000' in text:
          print(f'[{f}] PAGE {i+1}')
          print(text)
  except: pass
