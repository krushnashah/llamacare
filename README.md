# Llamacare: A Locally Run RAG-Based Healthcare Chatbot for PCOS


Llamacare is a privacy-focused Retrieval-Augmented Generation (RAG) chatbot designed to provide AI-driven responses to PCOS-related healthcare queries. By integrating a locally deployed architecture, it ensures data privacy while offering evidence-based medical insights.

**This project is actively being updated. Expect frequent improvements and optimizations.** 

---

## **Features**
- **Privacy-Focused**: Runs locally, ensuring patient data is not shared with external servers.
- **Retrieval-Augmented Generation (RAG)**: Uses FAISS (also experimenting with ChromaDB) for fast and efficient document retrieval.
- **PubMed Central Integration**: Automatically curates medical literature via the NCBI Entrez API.
- **Powered by LLaMA 3.2**: Generates high-quality, context-aware medical responses.
- **Optimized Retrieval Pipeline**: Enhances search accuracy for better healthcare insights.

---

## **Installation Guide**

### **Prerequisites**
- Python **3.12**
- Git
- **Pipenv** (install via `pip install pipenv`)

### **Clone the Repository**
```bash
git clone https://github.com/krushnashah/llamacare.git
cd llamacare
```

### **Setting Up the Virtual Environment**
Llamacare uses **Pipenv** for dependency management. Run the following command to install all required dependencies:
```bash
pipenv install
```

This will:
- Create a virtual environment if one does not exist.
- Install dependencies from `Pipfile.lock` (ensuring the same package versions across environments).

### **Activating the Virtual Environment**
Before running the application, activate the Pipenv shell:
```bash
pipenv shell
```

### **Running Llamacare Locally**
After activating the Pipenv shell, start the chatbot with:
```bash
streamlit run app.py
```
This will launch the Llamacare interface on a local web server. Open the provided URL in your browser to interact with the chatbot.

If you want to exit the virtual environment at any point, simply type:
```bash
exit
```

---

## **Dependency Management**

### **Installing New Dependencies**
To install additional packages:
```bash
pipenv install package-name
```

### **Uninstalling a Dependency**
To remove an installed package:
```bash
pipenv uninstall package-name
```

### **Updating Dependencies**
If dependencies are updated in the `Pipfile` or `Pipfile.lock`, you can apply the updates with:
```bash
pipenv update
```

---

## **Project Status & Future Updates**
Llamacare is actively being developed. Some planned enhancements include:
- Improved document retrieval using **ChromaDB** instead of FAISS.
- Support for **multimodal inputs** (e.g., text and images for better medical guidance).
- UI improvements to provide a better user experience.

You can track upcoming changes in the **Issues** and **Pull Requests** sections of this repository.


---

## **License**
This project is **private and proprietary**. If you wish to use it, please contact the repository owner.

---





