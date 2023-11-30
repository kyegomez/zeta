find ./tests -name "*.py" -type f | while read file
do
  filename=$(basename "$file")
  dir=$(dirname "$file")
  mv "$file" "$dir/test_$filename"
done