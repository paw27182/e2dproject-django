/*-------------------
 login by ENTER key
-------------------*/
$(function () {
	$("#inputPassword").keypress(function (e) {
		if (e.keyCode == 13) {
			var username = $('#inputUsername').val();  // unused
			var password = $('#inputPassword').val();  // unused

			id = $(this).parent().attr("id");
			if (id != "form_button_for_login") {  // in case Sign Up or Change Password
				return;
			};

			$.ajax({
				url: '/login',
				data: $('form').serialize(),  // username, password
				type: 'POST',
			})
				.done(function (output, status, xhr) {
					console.log(output);
				})
				.fail(function (error) {
					console.log(error);
				});  // ajax
		};  // if
	});  // keypress
});  // function
