const input = document.getElementById('case_file');
const fileName = document.getElementById('file_name');
if (input && fileName) {
  input.addEventListener('change', () => {
    fileName.textContent = input.files.length ? `Selected: ${input.files[0].name}` : '';
  });
}
