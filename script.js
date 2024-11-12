async function submitForm() {
    const inputData = {
        Pregnancies: parseInt(document.getElementById("Pregnancies").value),
        Glucose: parseFloat(document.getElementById("Glucose").value),
        BloodPressure: parseFloat(document.getElementById("BloodPressure").value),
        SkinThickness: parseFloat(document.getElementById("SkinThickness").value),
        Insulin: parseFloat(document.getElementById("Insulin").value),
        BMI: parseFloat(document.getElementById("BMI").value),
        DiabetesPedigreeFunction: parseFloat(document.getElementById("DiabetesPedigreeFunction").value),
        Age: parseInt(document.getElementById("Age").value)
    };

    try {
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(inputData)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const result = await response.json();
        document.getElementById("result").innerText = `Result: ${result.diabetes_result}`;
    } catch (error) {
        console.error("Error:", error);
        document.getElementById("result").innerText = "An error occurred. Please try again.";
    }
}
