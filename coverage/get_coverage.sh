source_dir="$1"
echo "source_dir: $source_dir"
tests_dir="$2"
echo "test_dir: $tests_dir"
cd "$source_dir"
pwd
find . -name "*.c" | while read cfile; do 
    echo $cfile
    rm "a.out"; 
    gcc $cfile -lm -w -fprofile-arcs -ftest-coverage -o "a.out";
    find "$tests_dir" -name "IN_*.txt" | while read testin; do

    (ulimit -t 3; ./a.out < $testin >> /dev/null;)

    cfile_name=${cfile/"./"/""}
    cfile_name=${cfile_name/".c"/""}
    gcov "$cfile"
    mv "$cfile.gcov" "$testin-$cfile_name.gcov"
    rm "$cfile_name.gcda"
done;
done;
