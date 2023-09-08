function deleteNote(noteId) {
    fetch("/delete-note", {
      method: "POST",
      body: JSON.stringify({ noteId: noteId }),
    }).then((_res) => {
      window.location.href = "/notes";
    });
  }

  function deleteImage(imageId) {
    fetch("/delete-image", {
      method: "POST",
      body: JSON.stringify({ imageId: imageId }),
    }).then((_res) => {
      window.location.href = "/notes";
    });
  }