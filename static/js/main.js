$(document).ready(function () {
  // Init
  $(".image-section").hide();
  $(".newl").hide();
  $("#result").hide();
  //$(".after-otp").hide();

  // Upload Preview
  function readURL(input) {
    if (input.files && input.files[0]) {
      var reader = new FileReader();
      reader.onload = function (e) {
        $("#imagePreview").css(
          "background-image",
          "url(" + e.target.result + ")"
        );
        $("#imagePreview").css("background-size", "450px 450px");
        $("#imagePreview").hide();
        $("#imagePreview").fadeIn(650);
      };
      reader.readAsDataURL(input.files[0]);
    }
  }

  //$(".btn-login").click(function validate_OTP() {
  //$(".after-otp").show(2000)
  //});

  $(".imageUpload").change(function () {
    $("#imagehidden").hide();
    $(".image-section").show();
    $("#btn-predict").css("margin-left", "240px");
    $("#result").css("margin-left", "150px");
    $("#result").css("width", "400px");
    $("#btn-predict").show();
    $("#result").text("");
    $("#result").hide();
    readURL(this);
  });

  // Predict
  $("#btn-predict").click(function () {
    var form_data = new FormData($("#upload-file")[0]);

    // Show loading animation
    $(this).hide();
    $(".newl").show();

    // Make prediction by calling api /predict
    $.ajax({
      type: "POST",
      url: "/predict",
      data: form_data,
      contentType: false,
      cache: false,
      processData: false,
      async: true,
      success: function (data) {
        // Get and display the result
        $(".newl").hide();
        $("#result").fadeIn(600);
        $("#result").text(" Result:  " + data);
        console.log("Success!");
      },
    });
  });
});

//$(document).on("submit", "#login-number", function (e) {
// e.preventDefault();
// console.log("Hey I am triggered");
//});
