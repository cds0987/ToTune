from flask import Blueprint, render_template
from functools import wraps
from flask import Blueprint, render_template, request, redirect, url_for, session, jsonify
import json
import os
mainpage_bth = Blueprint('mainpage', __name__)
@mainpage_bth.route("/")
def usermainpage():
    # Render the template for the user main page
    return render_template("index.html")

@mainpage_bth.route("/models")
def models():
    # Render the template for the user main page
    return render_template("SupportModel.html")


@mainpage_bth.route("/CoreTuneSetUp")
def finetune():
    return render_template("CoreTuneSetUp.html")


@mainpage_bth.route("/load")
def load():
    return render_template("Load.html")

@mainpage_bth.route("/finetunesetup")
def work():
    return render_template("Finetunesetup.html")

@mainpage_bth.route("/dataset")
def dataset():
    return render_template("Dataset.html")



CSV_PATH = "static/data/linkedin-company-information.csv"
import csv
import random
def format_row(row):
    """Merge all column values into one string"""
    return ", ".join(
        f"{k} {v}" for k, v in row.items() if v and v.strip()
    )

def load_csv():
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = reader.fieldnames
    return rows, columns
@mainpage_bth.route("/showsamples")
def get_samples():
    shuffle = request.args.get("shuffle", "false") == "true"

    rows, columns = load_csv()

    if shuffle:
        random.shuffle(rows)

    # one sample only
    sample_row = rows[0]

    # convert row → dict {column: value}
    sample_dict = sample_row

    return jsonify({
        "columns": columns,
        "sample": sample_dict
    })



@mainpage_bth.route("/Tunning")
def tunning():
    # Render the template for the user main page
    return render_template("Tunning.html")




from Webapp.User import update_user_json
@mainpage_bth.route("/select-model", methods=["POST"])
def select_model():
    data = request.get_json()

    update_user_json({
        "workmodel": data.get("workmodel"),
        "mode": data.get("mode")
    })

    if data.get("mode") == "finetuning":
        return jsonify(status="ok", redirect="/CoreTuneSetUp")

    return jsonify(status="ok")

@mainpage_bth.route("/select-finetune", methods=["POST"])
def select_tunes():
    data = request.get_json()

    update_user_json({
        "Task": data.get("Task"),
        "TuneModes": data.get("TuneModes")
    })

    return jsonify(status="ok")


@mainpage_bth.route("/select-data", methods=["POST"])
def select_data():
    data = request.get_json()

    update_user_json({
        "Data Access": data.get("access"),
        "Platform": data.get("platform")
    })

    return jsonify(status="ok")

@mainpage_bth.route("/select-setup", methods=["POST"])
def select_setup():
    data = request.get_json()
    update_user_json({
        "Task": data.get("Task"),
        "TuneModes": data.get("TuneModes")
    })
    update_user_json({
        "Data Access": data.get("access"),
        "Platform": data.get("platform")
    })
    update_user_json({
        "Waitting": True
    })
    return jsonify(status="ok")




from time import sleep
@mainpage_bth.route("/load", methods=["POST"])
def load_():
    sleep(3)
    update_user_json({
        "LoadModel": True,
    })
    update_user_json({
        "LoadData": True,
    })    
    update_user_json({
        "Waitting": False,
    })
    return jsonify(status="ok")
    
@mainpage_bth.route("/status", methods=["GET"])
def status():
    with open("static/data/user.json") as f:
        data = json.load(f)

    return jsonify({
        "Waitting": data.get("Waitting", True),
        "LoadModel": data.get("LoadModel", False),
        "LoadData": data.get("LoadData", False),
        "Training": training_state["done"]
    })



@mainpage_bth.route("/api/vram-calc", methods=["POST"])
def vram_calc():
    data = request.get_json()

    # ---------- BASIC INPUT ----------
    batch = int(data.get("batch", 8))
    seq = int(data.get("seq", 1024))
    grad = int(data.get("grad", 1))
    options = data.get("options", {})

    model = data.get("model")
    task = data.get("task")
    mode = data.get("mode")
    adaptation = data.get("adaptation")  # 👈 NEW

    # ---------- BASE VRAM (FULL FINETUNE) ----------
    base_model_mem = 8.0  # GB (model weights)
    activation_mem = batch * seq * 0.000002
    grad_mem = grad * 0.5

    # ---------- OPTIMIZATION FLAGS ----------
    if options.get("flash"):
        activation_mem *= 0.7
    if options.get("checkpoint"):
        activation_mem *= 0.5
    if options.get("offload"):
        grad_mem *= 0.6

    # ---------- ADAPTATION LOGIC (TEST ONLY) ----------
    peft_mem = 0.0

    if adaptation:
        method = adaptation.get("method")
        rank = int(adaptation.get("rank", 8))
        module = adaptation.get("module", "full")

        # Reduce base model memory (frozen weights)
        base_model_mem *= 0.4

        # Simple PEFT memory formula
        peft_mem = rank * 0.02  # GB per rank (FAKE, FOR TEST)

        if method == "ia3":
            peft_mem *= 0.5
        elif method == "adalora":
            peft_mem *= 0.8

        if module == "attention":
            peft_mem *= 0.6
        elif module == "mlp":
            peft_mem *= 0.7

    # ---------- TOTAL VRAM ----------
    total_vram = round(
        base_model_mem + activation_mem + grad_mem + peft_mem,
        2
    )

    capacity = 24
    percent = min(100, int((total_vram / capacity) * 100))
    trainspeed = 200
    sample_sec = 0.06
    step_sec = 2
    update_user_json({
        "vram_used": total_vram,
        'trainspeed': trainspeed,
        'sample_sec': sample_sec,
        'step_sec': step_sec
    })
    # ---------- RESPONSE ----------
    return jsonify({
        "vram_used": total_vram,
        "vram_total": capacity,
        "percent": percent,
        "trainspeed": trainspeed,
        "sample_sec": sample_sec,
        "step_sec": step_sec,
        "status": "OK" if total_vram <= capacity else "INSUFFICIENT",

        # EXTRA (mock metrics)
        "power": round(180 + batch * 1.3, 1),
        "cost": round(0.02 + batch * 0.0005 + peft_mem * 0.01, 3),
        "co2": round(1.8 + batch * 0.02, 2),

        # DEBUG (REMOVE LATER)
        "debug": {
            "mode": mode,
            "adaptation": adaptation,
            "base_model_mem": round(base_model_mem, 2),
            "activation_mem": round(activation_mem, 2),
            "grad_mem": round(grad_mem, 2),
            "peft_mem": round(peft_mem, 2)
        }
    })


@mainpage_bth.route("/saveconfig", methods=["POST"])
def save_config():
    data = request.json
    update_user_json({
        "adaptation": data['adaptation'],
        "options" : data['options']
    })
    return jsonify({"success": True})

@mainpage_bth.route("/savedata", methods=["POST"])
def savedata():
    data = request.json  # { "columns_data": [...] }
    print(data)

    update_user_json({
        "columns_data": data.get("columns_data", [])
    })
    update_user_json({
        "Training": False,
    })
    return jsonify({"success": True})

import random
training_state = {
    "epochs": [],
    "loss": [],
    "accuracy": [],
    "f1": [],
    "done": False
}

@mainpage_bth.route("/lossdata")
def lossdata():
    if not training_state["done"]:
        epoch = len(training_state["epochs"]) + 1

        if epoch <= 30:
            training_state["epochs"].append(epoch)

            # simulate metrics
            loss = training_state["loss"][-1] if training_state["loss"] else 1.0
            loss -= random.uniform(0.01, 0.05)
            loss = max(loss, 0.05)

            acc = training_state["accuracy"][-1] if training_state["accuracy"] else 0.55
            acc += random.uniform(0.01, 0.03)
            acc = min(acc, 0.99)

            f1 = training_state["f1"][-1] if training_state["f1"] else 0.5
            f1 += random.uniform(0.01, 0.03)
            f1 = min(f1, 0.99)

            training_state["loss"].append(round(loss, 4))
            training_state["accuracy"].append(round(acc, 4))
            training_state["f1"].append(round(f1, 4))

        if epoch >= 30:
            training_state["done"] = True

    return jsonify(training_state)