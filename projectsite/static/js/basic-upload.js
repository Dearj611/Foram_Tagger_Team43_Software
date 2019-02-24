$(function () {

  $(".js-upload-photos").click(function () {
    $("#fileupload").click();
  });

  $("#fileupload").fileupload({
    dataType: 'html',
    done: function (e, data) {
      location.reload();
    }
  });

  $('#edit_button').click(function() {
      $('.edit').toggle("slide");
    });

  $("#myBtn").click(function(){
    $("#myModal").modal();
  });

});
