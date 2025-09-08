import sqlite3
import pandas as pd
import joblib
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
import os

app = Flask(__name__)
app.secret_key = "a_very_secure_secret_key"


try:
    print("🔄 Loading application artifacts...")
    model = joblib.load("diet_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    column_order = joblib.load("column_order.pkl")
    food_db_path = os.path.join("datasets", "Raw_dataset", "food_recommendations.csv")
    food_df = pd.read_csv(food_db_path) 
    
    print("✅ All artifacts loaded successfully.")
    artifacts_loaded = True
except FileNotFoundError as e:
    print(f"❌ Critical Error: Missing a required file -> {e}")
    print("🚨 Please run train_model.py to generate model files and ensure food_recommendations.csv is in the correct folder.")
    artifacts_loaded = False


def init_db():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS appointments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doctor_name TEXT NOT NULL,
            patient_name TEXT NOT NULL,
            age INTEGER,
            contact TEXT,
            date TEXT,
            time TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()


@app.route("/")
def index():
    return render_template("home.html")

@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = generate_password_hash(request.form["password"])
        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            flash("✅ Registration successful. Please login.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("⚠️ Username already exists!", "danger")
        finally:
            conn.close()
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username=?", (username,))
        user = cursor.fetchone()
        conn.close()
        if user and check_password_hash(user[2], password):
            session["user"] = username
            flash("✅ Login successful!", "success")
            return redirect(url_for("home"))
        else:
            flash("⚠️ Invalid credentials", "danger")
    return render_template("login.html")

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        flash("⚠️ Please login first", "danger")
        return redirect(url_for("login"))
    return render_template("dashboard.html", username=session["user"])

@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("✅ Logged out successfully", "success")
    return redirect(url_for("login"))

@app.route("/get_recommendation")
def get_recommendation():
    return render_template("recommendation_form.html")


@app.route("/predict", methods=["POST"])
def predict():
    if not artifacts_loaded:
        flash("❌ Server Error: Application files are not loaded. Cannot make a recommendation.", "danger")
        return redirect(url_for("get_recommendation"))
    try:
        user_data = {
            'Age': float(request.form.get("age")),
            'Gender': request.form.get("gender"),
            'Weight_kg': float(request.form.get("weight")),
            'Height_cm': float(request.form.get("height")),
            'Occupation': request.form.get("occupation"),
            'Physical_Activity_Level': request.form.get("activity"),
            'Sleep_Duration': float(request.form.get("sleep_duration")),
            'Stress_Level': float(request.form.get("stress_level")),
            'Alcohol_Consumption': request.form.get("alcohol_consumption"),
        }
        allergies = request.form.get("allergies")
        height_m = user_data['Height_cm'] / 100
        if height_m <= 0:
            flash("⚠️ Height must be a positive number.", "danger")
            return redirect(url_for("get_recommendation"))
        user_data['BMI'] = round(user_data['Weight_kg'] / (height_m ** 2), 2)
        prediction_input = user_data.copy()
        for col, le in label_encoders.items():
            if col in prediction_input:
                try:
                    prediction_input[col] = le.transform([str(prediction_input[col])])[0]
                except ValueError:
                    prediction_input[col] = 0 
        input_df = pd.DataFrame([prediction_input], columns=column_order)
        input_scaled = scaler.transform(input_df)
        predicted_category = model.predict(input_scaled)[0]
        plan_details = food_df[food_df['Diet_Recommendation'] == predicted_category]

        if plan_details.empty:
            flash("Could not find a detailed plan for the predicted category.", "warning")
            plan_details = food_df[food_df['Diet_Recommendation'] == "Adult - Healthy Weight"].iloc[0]
        else:
            plan_details = plan_details.iloc[0]
        final_result = {
            'type': predicted_category,
            'nutrient_focus': plan_details.get('Nutrient_Focus'),
            'activity': plan_details.get('Activity_Recommendation'),
            'vegetables': plan_details.get('Vegetable_Intake'),
            'fruits': plan_details.get('Fruit_Intake'),
            'protein': plan_details.get('Protein_Intake'),
            'carbs': plan_details.get('Carbohydrate_Intake'),
            'advice': plan_details.get('General_Advice'),
        }
        final_result = refine_with_allergies(final_result, allergies)

        return render_template("recommendation.html", result=final_result, bmi=user_data['BMI'], age=int(user_data['Age']))
    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        flash("⚠️ An error occurred during prediction. Please check your inputs.", "danger")
        return redirect(url_for("get_recommendation"))
@app.route("/doctor_info")
def doctor_info():
    doctors = [
        {"id": 1, "name": "Dr. Ananya Sharma", "specialty": "Cardiologist", "image": "https://placehold.co/120x120/E0F7FA/00796B?text=Dr.+S"},
        {"id": 2, "name": "Dr. Ravi Verma", "specialty": "Dietitian / Nutritionist", "image": "https://placehold.co/120x120/FFF3E0/E65100?text=Dr.+V"},
        {"id": 3, "name": "Dr. Priya Desai", "specialty": "Endocrinologist", "image": "https://placehold.co/120x120/F3E5F5/6A1B9A?text=Dr.+D"},
        {"id": 4, "name": "Dr. Arjun Mehta", "specialty": "Dermatologist", "image": "https://placehold.co/120x120/E8F5E9/2E7D32?text=Dr.+M"},
        {"id": 5, "name": "Dr. Sneha Patel", "specialty": "General Physician", "image": "https://placehold.co/120x120/EDE7F6/512DA8?text=Dr.+P"},
        {"id": 6, "name": "Dr. Vikram Singh", "specialty": "Pediatrician", "image": "https://placehold.co/120x120/E1F5FE/0277BD?text=Dr.+S"},
    ]
    return render_template("doctor_info.html", doctors=doctors)

@app.route("/doctor_details/<int:doctor_id>")
def doctor_details(doctor_id):
    mock_doctors_db = {
        1: {"id": 1, "name": "Dr. Ananya Sharma", "specialty": "Cardiologist", "image": "https://placehold.co/120x120/E0F7FA/00796B?text=Dr.+S", "about": "Expert in heart health and preventive care.", "qualifications": "MD, FACC"},
        2: {"id": 2, "name": "Dr. Ravi Verma", "specialty": "Dietitian / Nutritionist", "image": "https://placehold.co/120x120/FFF3E0/E65100?text=Dr.+V", "about": "Specializes in creating personalized diet plans.", "qualifications": "M.Sc. - Dietetics"},
        3: {"id": 3, "name": "Dr. Priya Desai", "specialty": "Endocrinologist", "image": "https://placehold.co/120x120/F3E5F5/6A1B9A?text=Dr.+D", "about": "Focuses on diabetes and thyroid disorders.", "qualifications": "DM - Endocrinology"},
        4: {"id": 4, "name": "Dr. Arjun Mehta", "specialty": "Dermatologist", "image": "https://placehold.co/120x120/E8F5E9/2E7D32?text=Dr.+M", "about": "Specialist in skin health and cosmetic dermatology.", "qualifications": "MD - Dermatology"},
        5: {"id": 5, "name": "Dr. Sneha Patel", "specialty": "General Physician", "image": "https://placehold.co/120x120/EDE7F6/512DA8?text=Dr.+P", "about": "Provides comprehensive primary care for all ages.", "qualifications": "MBBS, MD"},
        6: {"id": 6, "name": "Dr. Vikram Singh", "specialty": "Pediatrician", "image": "https://placehold.co/120x120/E1F5FE/0277BD?text=Dr.+S", "about": "Dedicated to the health and well-being of children.", "qualifications": "MD - Pediatrics"},
    }
    doctor = mock_doctors_db.get(doctor_id)
    return render_template("doctor_details.html", doctor=doctor)

@app.route('/book_appointment', methods=['POST'])
def book_appointment():
    doctor_name = request.form['doctor_name']
    patient_name = request.form['patient_name']
    age = request.form['age']
    contact = request.form['contact']
    date = request.form['date']
    time = request.form['time']
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO appointments (doctor_name, patient_name, age, contact, date, time) VALUES (?, ?, ?, ?, ?, ?)',
                   (doctor_name, patient_name, age, contact, date, time))
    conn.commit()
    conn.close()
    flash('✅ Your appointment has been successfully booked!', 'success')
    return redirect(url_for('doctor_info'))
@app.route("/recipe")
def recipe():
    recipes_data = {
        'quinoa_salad': {"name": "Quick Quinoa Salad", "description": "A light, refreshing, and protein-packed salad.", "video_id": "cREMk8-eT8A"},
        'chicken_veggies': {"name": "Sheet Pan Chicken & Veggies", "description": "An easy one-pan meal for busy weeknights.", "video_id": "vS3_RclU_vU"},
        'lentil_soup': {"name": "Hearty Lentil Soup", "description": "A warm, comforting, and fiber-rich soup.", "video_id": "g-5ktm_f-sI"},
        'berry_smoothie': {"name": "Antioxidant Berry Smoothie", "description": "A delicious smoothie to start your day.", "video_id": "ASi-gSAx9F4"},
        'baked_salmon': {"name": "Garlic Herb Baked Salmon", "description": "A simple dish rich in Omega-3 fatty acids.", "video_id": "rJg_26j_pM0"},
        'veggie_stir_fry': {"name": "Vegetable Stir-Fry", "description": "A colorful stir-fry you can customize.", "video_id": "3yX3r0f9yPA"}
    }
    return render_template("recipe.html", recipes=recipes_data)

@app.route("/recipe_details/<recipe_id>")
def recipe_details(recipe_id):
    recipes_data = {
        'quinoa_salad': {"name": "Quick Quinoa Salad", "video_id": "cREMk8-eT8A"},
        'chicken_veggies': {"name": "Sheet Pan Chicken & Veggies", "video_id": "vS3_RclU_vU"},
        'lentil_soup': {"name": "Hearty Lentil Soup", "video_id": "g-5ktm_f-sI"},
        'berry_smoothie': {"name": "Antioxidant Berry Smoothie", "video_id": "ASi-gSAx9F4"},
        'baked_salmon': {"name": "Garlic Herb Baked Salmon", "video_id": "rJg_26j_pM0"},
        'veggie_stir_fry': {"name": "Vegetable Stir-Fry", "video_id": "3yX3r0f9yPA"}
    }
    recipe = recipes_data.get(recipe_id)
    if not recipe:
        flash("Recipe not found!", "danger")
        return redirect(url_for('recipe'))
    return render_template("recipe_details.html", recipe=recipe)

def refine_with_allergies(base_result, allergies: str):
    if allergies and allergies.strip():
        allergy_list = [item.strip().title() for item in allergies.split(',')]
        refinement_message = f"⚠️ Allergy Alert: Remember to avoid these ingredients: {', '.join(allergy_list)}."
        base_result['refinements'] = [refinement_message]
    return base_result

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')

