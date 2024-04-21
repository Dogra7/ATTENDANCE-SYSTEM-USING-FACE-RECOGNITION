document.getElementById('capture_button').addEventListener('click', function() {
    // Send an AJAX request to the server to fetch the recognized face name
    var xhr = new XMLHttpRequest();
    xhr.open('GET', '/get_recognized_face_name', true);
    xhr.onload = function () {
        if (xhr.status === 200) {
            var response = JSON.parse(xhr.responseText);
            var recognizedName = response.recognized_face_name;
            if (recognizedName !== "Unknown") {
                // Send an AJAX request to the server to store the recognized face name in the database
                var xhrStore = new XMLHttpRequest();
                xhrStore.open('GET', '/store_recognized_face', true);
                xhrStore.send();

                // Display the recognized face name in an alert box
                alert("Marked Attendance For: " + recognizedName);
                // Redirect to the home page after the user clicks "OK"
                window.location.href = "/home";
            } else {
                // Display "Face not recognized" in an alert box
                alert("Face not recognized");
            }
        } else {
            alert('Request failed. Please try again later.');
        }
    };
    xhr.send();
});