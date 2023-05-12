<!DOCTYPE html>
<html lang="en">

<head>
  <meta charset="UTF-8">
  <meta http-equiv="X-UA-Compatible" content="IE=edge">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="stylesheet" href="./src/css/error_404.css">
  <title>Error 404</title>
</head>

<body>
  <?php require('./components/header.php'); ?>
  <div class="py-6 sm:py-8 lg:py-12 main flex justify-center ">
    <div class="max-w-screen-2xl px-4 md:px-8 mx-auto flex items-center">
      <div class="flex flex-col justify-center items-center">
        <!-- logo - start -->
        <a href="./index" class="inline-flex items-center logo text-2xl md:text-3xl font-bold gap-2.5 mb-8" aria-label="logo">          
        <img class="h-10 dark_logo" src="https://cdn.discordapp.com/attachments/972503652310843552/1005902444624105502/logo_dark.png">
        <img class="h-10 light_logo" src="https://cdn.discordapp.com/attachments/972503652310843552/1005902445282603028/logo.png">
          
          Artifical Doctor
        </a>
        <!-- logo - end -->

        <p class="text-sm md:text-base font-semibold uppercase mb-4 main-txt">That’s a 404</p>
        <h1 class="text-2xl md:text-3xl font-bold text-center mb-2">Page not found</h1>

        <p class="max-w-screen-md txt md:text-lg text-center mb-12">The page you’re looking for doesn’t exist.</p>

          <a href="./index" class="inline-block home_btn focus-visible:ring ring-indigo-300 text-sm md:text-base font-semibold text-center rounded-lg outline-none transition duration-100 px-8 py-3">Go home</a>
      </div>
    </div>
  </div>
  <?php require('./components/footer.php'); ?>
</body>

</html>