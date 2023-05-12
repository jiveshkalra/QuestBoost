<!DOCTYPE html>
<html lang="en">
<?php
function duplicateMessage($username)
{
  echo '
  <div class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mt-20" role="alert" id="to_hide">
    <strong class="font-bold">Duplicate username!</strong>
    <span class="block sm:inline">An account with the username(' . $username . ') already exist,Please try <a href= "./login.php"><p class="inline-block text-blue-500 font-semibold">login</p></a></span>
      <span class="absolute top-0 bottom-0 right-0 px-4 py-3">
        <svg class="fill-current h-6 w-6 text-red-500"onclick="hide()"role="button" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20"><title>Close</title><path d="M14.348 14.849a1.2 1.2 0 0 1-1.697 0L10 11.819l-2.651 3.029a1.2 1.2 0 1 1-1.697-1.697l2.758-3.15-2.759-3.152a1.2 1.2 0 1 1 1.697-1.697L10 8.183l2.651-3.031a1.2 1.2 0 1 1 1.697 1.697l-2.758 3.152 2.758 3.15a1.21.2 0 0 1 0 1.698z"/></svg>
      </span>
  </div>
';
}
function duplicateEmail($email)
{
  echo '
  <div class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mt-20" role="alert" id="to_hide">
  <strong class="font-bold">Duplicate email!</strong>
  <span class="block sm:inline">An account with the username(' . $email . ') already exist,Please try <a href= "./login.php"><p class="inline-block text-blue-500 font-semibold">login</p></a></span>
  <span class="absolute top-0 bottom-0 right-0 px-4 py-3">
  <svg class="fill-current h-6 w-6 text-red-500" onclick="hide()" role="button" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20"><title>Close</title><path d="M14.348 14.849a1.2 1.2 0 0 1-1.697 0L10 11.819l-2.651 3.029a1.2 1.2 0 1 1-1.697-1.697l2.758-3.15-2.759-3.152a1.2 1.2 0 1 1 1.697-1.697L10 8.183l2.651-3.031a1.2 1.2 0 1 1 1.697 1.697l-2.758 3.152 2.758 3.15a1.21.2 0 0 1 0 1.698z"/></svg>
  </span>
  </div>
';
}
function no_err()
{
  echo '
  <div class="bg-lime-100 border border-lime-400 text-lime-700 px-4 py-3 rounded relative mt-20" role="alert" id="to_hide">
  <strong class="font-bold">SUCCESS!</strong>
  <span class="block sm:inline">Your account was made successfully!</span>
  <span class="absolute top-0 bottom-0 right-0 px-4 py-3">
  <svg class="fill-current h-6 w-6 text-lime-500" onclick="hide()" role="button" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20"><title>Close</title><path d="M14.348 14.849a1.2 1.2 0 0 1-1.697 0L10 11.819l-2.651 3.029a1.2 1.2 0 1 1-1.697-1.697l2.758-3.15-2.759-3.152a1.2 1.2 0 1 1 1.697-1.697L10 8.183l2.651-3.031a1.2 1.2 0 1 1 1.697 1.697l-2.758 3.152 2.758 3.15a1.21.2 0 0 1 0 1.698z"/></svg>
  </span>
  </div>';
}
function servers_down()
{
  echo ' 
  <div class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mt-20" role="alert" id="to_hide">
  <strong class="font-bold">Some error occured in regiseration</strong>
  <span class="block sm:inline">Probably servers are down ,please try again later or contact website admins</span>
  <span class="absolute top-0 bottom-0 right-0 px-4 py-3">
  <svg class="fill-current h-6 w-6 text-red-500" role="button" onclick="hide()" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20"><title>Close</title><path d="M14.348 14.849a1.2 1.2 0 0 1-1.697 0L10 11.819l-2.651 3.029a1.2 1.2 0 1 1-1.697-1.697l2.758-3.15-2.759-3.152a1.2 1.2 0 1 1 1.697-1.697L10 8.183l2.651-3.031a1.2 1.2 0 1 1 1.697 1.697l-2.758 3.152 2.758 3.15a1.21.2 0 0 1 0 1.698z"/></svg>
  </span>
  </div>';
}
?>

<head>
  <meta charset="UTF-8">
  <meta http-equiv="X-UA-Compatible" content="IE=edge">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.0/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-gH2yIJqKdNHPEq0n4Mqa/HGKIhSkIHeL5AyhkYV8i59U5AR6csBvApHHNl/vI1Bx" crossorigin="anonymous">
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" />
  <title>Daily Quester</title>
  <style>
    body {
      background: #073B4C;
    }

    .heading {
      color: #000;
    }

    .dark .heading {
      color: #c2185b;
    }
  </style>
</head>

<body>
  <?php require('components/header.php') ?>
  <div class="pt-6 sm:pt-8 lg:pt-12">
    <div class="max-w-screen-2xl px-4 md:px-8 mx-autoflex">
      <?php
        if (isset($_GET)) {
        echo $_GET['error'];
        if ($_GET['error'] == "duplicate_username") {
          duplicateMessage($_GET['username']);
        }
        elseif($_GET['error']=="duplicate_email"){
          duplicateEmail($_GET['email']);
        }
        elseif($_GET['error']=='none'){
          no_err();
        }
        elseif($_GET['error']=='servers_down'){
          servers_down();
        }
      } ?>
      <h2 class="heading text-2xl lg:text-3xl font-bold text-center mb-4 md:mb-8 mt-20">REGISTER</h2>
      <form class="max-w-lg border rounded-lg mx-auto" id="myform" style="margin-bottom: 150px;" action="./reg_backend" method='post'>
        <div class="flex flex-col gap-4 p-4 md:p-8 bg-white">
          <div>
            <label for="email" class="inline-block text-gray-800 text-sm sm:text-base mb-2">Email<span class="text-red-900 font-bold">*</span></label>
            <input name="email" type="email" id="email" placeholder="example@email.com" class="w-full bg-gray-50 text-gray-800 border focus:ring ring-indigo-300 rounded outline-none transition duration-100 px-3 py-2" required />
          </div>
          <div>
            <label for="username" class="inline-block text-gray-800 text-sm sm:text-base mb-2">Username<span class="text-red-900 font-bold">*</span></label>
            <input name="username" type="text" id="username" placeholder="Username" class="w-full bg-gray-50 text-gray-800 border focus:ring ring-indigo-300 rounded outline-none transition duration-100 px-3 py-2" required />
          </div>
          <div>
            <label for="password" class="inline-block text-gray-800 text-sm sm:text-base mb-2">Password<span class="text-red-900 font-bold">*</span></label>
            <input name="password" autocomplete="off" id="password" placeholder="PASSWORD@123" type='password' class="w-full bg-gray-50 text-gray-800 border focus:ring ring-indigo-300 rounded outline-none transition duration-100 px-3 py-2" required />
          </div>
          <div>
            <label for="code" class="inline-block text-gray-800 text-sm sm:text-base mb-2">Application Code</label>
            <input name="appCode" id="appCode" placeholder="1234" type="text" class="w-full bg-gray-50 text-gray-800 border focus:ring ring-indigo-300 rounded outline-none transition duration-100 px-3 py-2" />
          </div>
          <div id="app-codeHelp" class="form-text text-gray-500">You can change application code later or leave empty for now.</div>

          <button class="flex justify-center items-center bg-blue-500 hover:bg-blue-600 active:bg-blue-700 focus-visible:ring ring-blue-300 text-white text-sm md:text-base font-semibold text-center rounded-lg outline-none transition duration-100 gap-2 px-8 py-3" type='submit'>
            <span class="material-symbols-outlined">how_to_reg</span>Register
          </button>
        </div>
      </form>
    </div>
    <?php require('components/footer.php') ?>
</body>
<script>
  function hide(){
    document.getElementById('to_hide').style.display = 'none';
    window.location.replace('./register');
    }
</script>
</html>
<!-- // <div class="flex justify-center items-center relative">
//           <span class="h-px bg-gray-300 absolute inset-x-0"></span>
//           <span class="bg-white text-gray-400 text-sm relative px-4">Log in with social</span>
//         </div>

//         <button class="flex justify-center items-center bg-blue-500 hover:bg-blue-600 active:bg-blue-700 focus-visible:ring ring-blue-300 text-white text-sm md:text-base font-semibold text-center rounded-lg outline-none transition duration-100 gap-2 px-8 py-3">
//           <svg class="w-5 h-5 shrink-0" width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
//             <path d="M12 0C5.37273 0 0 5.37273 0 12C0 18.0164 4.43182 22.9838 10.2065 23.8516V15.1805H7.23764V12.0262H10.2065V9.92727C10.2065 6.45218 11.8996 4.92655 14.7878 4.92655C16.1711 4.92655 16.9025 5.02909 17.2489 5.076V7.82945H15.2787C14.0525 7.82945 13.6244 8.99182 13.6244 10.302V12.0262H17.2178L16.7302 15.1805H13.6244V23.8773C19.4815 23.0825 24 18.0747 24 12C24 5.37273 18.6273 0 12 0Z" fill="white" />
//           </svg>

//           Continue with Facebook
//         </button>

//         <button class="flex justify-center items-center bg-white hover:bg-gray-100 active:bg-gray-200 border border-gray-300 focus-visible:ring ring-gray-300 text-gray-800 text-sm md:text-base font-semibold text-center rounded-lg outline-none transition duration-100 gap-2 px-8 py-3">
//           <svg class="w-5 h-5 shrink-0" width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
//             <path d="M23.7449 12.27C23.7449 11.48 23.6749 10.73 23.5549 10H12.2549V14.51H18.7249C18.4349 15.99 17.5849 17.24 16.3249 18.09V21.09H20.1849C22.4449 19 23.7449 15.92 23.7449 12.27Z" fill="#4285F4" />
//             <path d="M12.2549 24C15.4949 24 18.2049 22.92 20.1849 21.09L16.3249 18.09C15.2449 18.81 13.8749 19.25 12.2549 19.25C9.12492 19.25 6.47492 17.14 5.52492 14.29H1.54492V17.38C3.51492 21.3 7.56492 24 12.2549 24Z" fill="#34A853" />
//             <path d="M5.52488 14.29C5.27488 13.57 5.14488 12.8 5.14488 12C5.14488 11.2 5.28488 10.43 5.52488 9.71V6.62H1.54488C0.724882 8.24 0.254883 10.06 0.254883 12C0.254883 13.94 0.724882 15.76 1.54488 17.38L5.52488 14.29Z" fill="#FBBC05" />
//             <path d="M12.2549 4.75C14.0249 4.75 15.6049 5.36 16.8549 6.55L20.2749 3.13C18.2049 1.19 15.4949 0 12.2549 0C7.56492 0 3.51492 2.7 1.54492 6.62L5.52492 9.71C6.47492 6.86 9.12492 4.75 12.2549 4.75Z" fill="#EA4335" />
//           </svg>

//           Continue with Google
//         </button>
//       </div>

//       <div class="flex justify-center items-center bg-gray-100 p-4">
//         <p class="text-gray-500 text-sm text-center">Don't have an account? <a href="#" class="text-indigo-500 hover:text-indigo-600 active:text-indigo-700 transition duration-100">Register</a></p>
//       </div> -->