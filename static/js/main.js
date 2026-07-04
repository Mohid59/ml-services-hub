// Small UI niceties: show chosen file name + spinner on submit.
document.addEventListener("DOMContentLoaded", function () {
  // Show selected filename next to file inputs.
  document.querySelectorAll('input[type="file"]').forEach(function (input) {
    input.addEventListener("change", function () {
      const label = document.querySelector('[data-filename-for="' + input.id + '"]');
      if (label) label.textContent = input.files.length ? input.files[0].name : "";
      // Image preview if present.
      const preview = document.querySelector('[data-preview-for="' + input.id + '"]');
      if (preview && input.files.length && input.files[0].type.startsWith("image/")) {
        preview.src = URL.createObjectURL(input.files[0]);
        preview.style.display = "block";
      }
    });
  });

  // Loading spinner: disable button + spin on form submit.
  document.querySelectorAll("form[data-run-form]").forEach(function (form) {
    form.addEventListener("submit", function () {
      const btn = form.querySelector('button[type="submit"]');
      if (btn) {
        btn.disabled = true;
        const sp = btn.querySelector(".spinner");
        if (sp) sp.style.display = "inline-block";
        const txt = btn.querySelector(".btn-text");
        if (txt) txt.textContent = "Running…";
      }
    });
  });
});
