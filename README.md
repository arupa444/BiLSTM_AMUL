
# BiLSTM_AMUL

## 🚀 Overview

**BiLSTM_AMUL** is a project that uses a Bidirectional Long Short-Term Memory (BiLSTM) neural network model for … *(briefly describe what your project does — e.g. classification / prediction / some domain-specific aim)*.  
The repository includes model definitions, sample data, and scripts to run training and inference — making it easier to reproduce experiments or build upon the code.

## 📂 Repository Structure

```

BiLSTM_AMUL/
│
├── models/                    # Contains model definitions / saved models
├── multiLayer.ipynb           # Notebook with experiments or demonstrations
├── app.py                     # Example script / main application (if applicable)
├── test.py                    # Test or inference script
├── requirements.txt           # Python dependencies
├── how.md                     # Documentation / instructions
├── modelsWeCanUSe.md          # Notes on alternate or usable models
├── MINIMUM DATA REQUIRED AND THE CONCLUSION.pdf  # Documentation / report
└── README.md                  # (this file)

````

You can expand or modify this as your project evolves.

## 📥 Installation & Setup

1. Clone the repository  
    ```bash
    git clone https://github.com/arupa444/BiLSTM_AMUL.git
    cd BiLSTM_AMUL
    ```

2. (Optional but recommended) Create a virtual environment  
    ```bash
    python3 -m venv venv
    source venv/bin/activate     # On Windows: `venv\Scripts\activate`
    ```

3. Install dependencies  
    ```bash
    pip install -r requirements.txt
    ```

4. (If your project needs some data) Prepare or download the required dataset according to instructions in `how.md` or `modelsWeCanUSe.md`.

## ▶️ Usage / Running the Code

Depending on what you want to do:

- **Run experiments / training**  
  Use the notebook `multiLayer.ipynb` (e.g. open in Jupyter) to run experiments, build and train the BiLSTM model, and observe results.  

- **Run scripts**  
  For the enpoint calling:  
  ```bash
  uvicorn app:app --reload
  ```
- **Run scripts**  
  For example, to run inference or test the model:  
  ```bash
  python test.py
  ```

* **Refer documentation**
  For more details on model choices, data requirements, usage instructions — check `how.md` or `modelsWeCanUSe.md`.

- **Run this json in swaggerUI**  
  For example, to run inference or test the model:  
  ```bash
      {
      "branchcode": "GC01",
      "materialcode": "SKU_A",
      "historical_data": [
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 500, "intransit_qty": 50, "pending_po_qty": 0, "lead_time_days": 7, "date": "2024-01-01"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 495, "intransit_qty": 50, "pending_po_qty": 0, "lead_time_days": 7, "date": "2024-01-02"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 490, "intransit_qty": 50, "pending_po_qty": 0, "lead_time_days": 7, "date": "2024-01-03"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 480, "intransit_qty": 0, "pending_po_qty": 0, "lead_time_days": 7, "date": "2024-01-04"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 475, "intransit_qty": 0, "pending_po_qty": 0, "lead_time_days": 7, "date": "2024-01-05"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 470, "intransit_qty": 0, "pending_po_qty": 50, "lead_time_days": 7, "date": "2024-01-06"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 460, "intransit_qty": 0, "pending_po_qty": 50, "lead_time_days": 7, "date": "2024-01-07"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 450, "intransit_qty": 0, "pending_po_qty": 50, "lead_time_days": 7, "date": "2024-01-08"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 445, "intransit_qty": 0, "pending_po_qty": 50, "lead_time_days": 7, "date": "2024-01-09"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 440, "intransit_qty": 0, "pending_po_qty": 50, "lead_time_days": 7, "date": "2024-01-10"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 435, "intransit_qty": 0, "pending_po_qty": 50, "lead_time_days": 7, "date": "2024-01-11"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 430, "intransit_qty": 50, "pending_po_qty": 0, "lead_time_days": 7, "date": "2024-01-12"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 480, "intransit_qty": 0, "pending_po_qty": 0, "lead_time_days": 7, "date": "2024-01-13"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 475, "intransit_qty": 0, "pending_po_qty": 0, "lead_time_days": 7, "date": "2024-01-14"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 470, "intransit_qty": 0, "pending_po_qty": 0, "lead_time_days": 7, "date": "2024-01-15"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 465, "intransit_qty": 0, "pending_po_qty": 0, "lead_time_days": 7, "date": "2024-01-16"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 460, "intransit_qty": 0, "pending_po_qty": 0, "lead_time_days": 7, "date": "2024-01-17"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 455, "intransit_qty": 0, "pending_po_qty": 0, "lead_time_days": 7, "date": "2024-01-18"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 450, "intransit_qty": 0, "pending_po_qty": 0, "lead_time_days": 7, "date": "2024-01-19"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 445, "intransit_qty": 0, "pending_po_qty": 0, "lead_time_days": 7, "date": "2024-01-20"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 440, "intransit_qty": 0, "pending_po_qty": 0, "lead_time_days": 7, "date": "2024-01-21"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 435, "intransit_qty": 0, "pending_po_qty": 0, "lead_time_days": 7, "date": "2024-01-22"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 430, "intransit_qty": 0, "pending_po_qty": 0, "lead_time_days": 7, "date": "2024-01-23"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 425, "intransit_qty": 0, "pending_po_qty": 0, "lead_time_days": 7, "date": "2024-01-24"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 420, "intransit_qty": 0, "pending_po_qty": 0, "lead_time_days": 7, "date": "2024-01-25"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 415, "intransit_qty": 0, "pending_po_qty": 0, "lead_time_days": 7, "date": "2024-01-26"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 410, "intransit_qty": 0, "pending_po_qty": 0, "lead_time_days": 7, "date": "2024-01-27"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 405, "intransit_qty": 0, "pending_po_qty": 0, "lead_time_days": 7, "date": "2024-01-28"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 400, "intransit_qty": 0, "pending_po_qty": 0, "lead_time_days": 7, "date": "2024-01-29"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 395, "intransit_qty": 0, "pending_po_qty": 0, "lead_time_days": 7, "date": "2024-01-30"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 390, "intransit_qty": 0, "pending_po_qty": 0, "lead_time_days": 7, "date": "2024-01-31"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 385, "intransit_qty": 0, "pending_po_qty": 0, "lead_time_days": 7, "date": "2024-02-01"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 380, "intransit_qty": 0, "pending_po_qty": 0, "lead_time_days": 7, "date": "2024-02-02"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 375, "intransit_qty": 0, "pending_po_qty": 0, "lead_time_days": 7, "date": "2024-02-03"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 370, "intransit_qty": 0, "pending_po_qty": 0, "lead_time_days": 7, "date": "2024-02-04"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 365, "intransit_qty": 0, "pending_po_qty": 0, "lead_time_days": 7, "date": "2024-02-05"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 360, "intransit_qty": 0, "pending_po_qty": 0, "lead_time_days": 7, "date": "2024-02-06"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 355, "intransit_qty": 0, "pending_po_qty": 0, "lead_time_days": 7, "date": "2024-02-07"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 350, "intransit_qty": 0, "pending_po_qty": 0, "lead_time_days": 7, "date": "2024-02-08"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 345, "intransit_qty": 0, "pending_po_qty": 0, "lead_time_days": 7, "date": "2024-02-09"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 340, "intransit_qty": 0, "pending_po_qty": 0, "lead_time_days": 7, "date": "2024-02-10"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 335, "intransit_qty": 0, "pending_po_qty": 0, "lead_time_days": 7, "date": "2024-02-11"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 330, "intransit_qty": 0, "pending_po_qty": 0, "lead_time_days": 7, "date": "2024-02-12"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 325, "intransit_qty": 0, "pending_po_qty": 0, "lead_time_days": 7, "date": "2024-02-13"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 320, "intransit_qty": 0, "pending_po_qty": 0, "lead_time_days": 7, "date": "2024-02-14"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 315, "intransit_qty": 0, "pending_po_qty": 0, "lead_time_days": 7, "date": "2024-02-15"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 310, "intransit_qty": 0, "pending_po_qty": 0, "lead_time_days": 7, "date": "2024-02-16"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 305, "intransit_qty": 0, "pending_po_qty": 0, "lead_time_days": 7, "date": "2024-02-17"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 300, "intransit_qty": 0, "pending_po_qty": 0, "lead_time_days": 7, "date": "2024-02-18"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 295, "intransit_qty": 0, "pending_po_qty": 0, "lead_time_days": 7, "date": "2024-02-19"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 290, "intransit_qty": 0, "pending_po_qty": 0, "lead_time_days": 7, "date": "2024-02-20"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 285, "intransit_qty": 0, "pending_po_qty": 0, "lead_time_days": 7, "date": "2024-02-21"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 280, "intransit_qty": 0, "pending_po_qty": 0, "lead_time_days": 7, "date": "2024-02-22"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 275, "intransit_qty": 0, "pending_po_qty": 0, "lead_time_days": 7, "date": "2024-02-23"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 270, "intransit_qty": 0, "pending_po_qty": 0, "lead_time_days": 7, "date": "2024-02-24"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 265, "intransit_qty": 0, "pending_po_qty": 0, "lead_time_days": 7, "date": "2024-02-25"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 260, "intransit_qty": 0, "pending_po_qty": 0, "lead_time_days": 7, "date": "2024-02-26"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 255, "intransit_qty": 0, "pending_po_qty": 0, "lead_time_days": 7, "date": "2024-02-27"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 250, "intransit_qty": 0, "pending_po_qty": 0, "lead_time_days": 7, "date": "2024-02-28"},
        {"branchcode": "GC01", "materialcode": "SKU_A", "stock_on_hand": 245, "intransit_qty": 0, "pending_po_qty": 0, "lead_time_days": 7, "date": "2024-02-29"}
      ]
    }
  ```

## 🧠 What is BiLSTM (and why use it)

BiLSTM stands for **Bidirectional Long Short-Term Memory** — a type of recurrent neural network architecture that processes sequences in both forward and backward directions, thereby capturing context from the past and future simultaneously.

This makes BiLSTM especially useful for tasks involving sequential data (e.g. text, time-series, etc.) — as it can learn dependencies from both previous and upcoming elements in a sequence.

## ✅ Features / What’s included

* BiLSTM model implementation and pre-defined model architecture
* Example notebook and demo code for training / evaluation
* Scripts for testing/inference (`test.py`, `app.py`)
* Documentation and notes (data requirements, model options)
* Easy to setup — dependencies listed in `requirements.txt`
