<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="X-UA-Compatible" content="IE=edge">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Register Backend</title>
</head>
<body>
  
<?php
require("./components/_dbconnect.php");
if ($_SERVER['REQUEST_METHOD'] == 'POST') {
  if ($_POST['email'] == '' || $_POST['username'] == '' || $_POST['password'] == '') {

  } else {
    $email = $_POST['email'];
    $username = $_POST['username'];
    $password = $_POST['password'];
    $appCode = $_POST['appCode'];
    $sql = "SELECT * FROM `all_users` WHERE `email` =  '$email' OR `username` = '$username'";
    $result = mysqli_query($conn, $sql);
    if ($result) {
      // If $result is greater than 0 then to user name or email has been taken already
      if (mysqli_num_rows($result) > 0) {
        // For already existing username
        $result_fetch = mysqli_fetch_assoc($result);
        if ($result_fetch['username'] == $username) {
          echo '<script>window.location.replace("./register?error=duplicate_username&username='.$username.'");</script>';
        }
        
        // For already existing email
        else if ($result_fetch['email'] == $email) {
          echo '<script>window.location.replace("./register?error=duplicate_email&email='.$email.'");</script>';
        }
      } else {
        $password = password_hash($password, PASSWORD_BCRYPT);
        // $verification_code = bin2hex(random_bytes(16));
        if ($appCode = '') {
          $appCode = '0000';
        }
        $sql = "INSERT INTO `all_users`(`email`, `username`,`password`,`appCode`) VALUES ('$email','$username','$password','$appCode')";
        echo $sql;
        // && send_mail($email, $verification_code,$referral_code_submit) For email
        if (mysqli_query($conn, $sql)) { 
          // If Query is run successfully , This code will run 
          // echo '<div class="container"><div class="alert alert-success" role="alert">
          // We have send a veriication email on the email id ' . $email . '<br>
          // Please check it once and get your account verified<br>
          // <strong>Sometimes it comes in spam mails to.</strong>
          // </div></div>';
          echo '<script>window.location.replace("./register?error=none");</script>';
        } else {
          // If Query is NOT run successfully , This code will run
          echo '<script>window.location.replace("./register?error=servers_down");</script>';
        }
      }
    } else {
      echo '<script>window.location.replace("./register?error=servers_down");</script>';
    }
  }
  // echo '<script> window.location.replace("./register")</script>';
} else {
  echo '<script>window.location.replace("./register")</script>';
}
?>
</body>
</html>
