$(function () {

  $(".js-upload-photos").click(function () {
    $("#fileupload").click();
  });

  $("#fileupload").fileupload({
    dataType: 'json',
    done: function (e, data) {
      if (data.result.is_valid) {
        console.log(data.result)
        new_location=window.location.href + '?imgLocation=' + data.result.url;
        console.log(new_location)
        window.location.replace(new_location);
      }
    }
  });

  $('#edit_button').click(function() {
      $('.edit').toggle("slide");
    });

  $("#myBtn").click(function(){
    $("#myModal").modal();
  });

});
