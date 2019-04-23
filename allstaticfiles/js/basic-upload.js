$(function () {

  $(".js-upload-photos").click(function () {
    $("#fileupload").click();
  });

  $("#fileupload").fileupload({
    dataType: 'json',
    done: function (e, data) {
      if (data.result.is_valid) {
        console.log(data.result)
        var queryParams = "";
        for (url of data.result.urls){
          if (window.location.href.includes('?') || queryParams.includes('?')){
            queryParams += '&imgLocation=' + url;
          }
          else{
            queryParams += '?imgLocation=' + url;
          }
        }
        console.log(queryParams);
        window.location.href += queryParams;
      }
    },
    'singleFileUploads': false
  });

  $('#edit_button').click(function() {
      $('.edit').toggle("slide");
    });

  $("#myBtn").click(function(){
    $("#myModal").modal();
  });

});
