$(function () {

  $(".js-upload-photos").click(function () {
    $("#fileupload").click();
  });

  $("#fileupload").fileupload({
    dataType: 'json',
    done: function (e, data) {
      if (data.result.is_valid) {
        $("#gallery tbody").append(
          "<tr><td><img src='" + data.result.url + "'><p>"+ data.result.species + "</p></td></tr>"
        )

        $("#display").append(
          "<div class='display_items'><img src='"+ data.result.url + "'style='width:100%'><p>"
          + data.result.species +"</p><button> Change Tag</button></div>"
        )
      }
    }
  });

});
