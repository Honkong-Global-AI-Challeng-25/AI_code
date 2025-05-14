import nbformat
from nbconvert import PythonExporter

ipynb_file = "lstm.ipynb"
output_py_file = "lstm.py"

with open(ipynb_file, "r", encoding="utf-8") as f:
    notebook = nbformat.read(f, as_version=4)

exporter = PythonExporter()
python_code, _ = exporter.from_notebook_node(notebook)

with open(output_py_file, "w", encoding="utf-8") as f:
    f.write(python_code)

print(f"✅ 변환 완료: {output_py_file}")

