# root_dir=$(pwd)
# root_dir="$root_dir/"
# echo "root_dir: $root_dir"

source_dir="$1"
echo "source_dir: $source_dir"
tests_dir="$2"

# # log_file="$root_dir/$prefix/$problem/coverage-log-file.txt"
echo $(date)
echo "Starting script"

compiled_binary="a.out"
captured_test_output="$prefix-$problem-test.out"
spaceless_actualtestout="$prefix-$problem-spaceless-test-out"

for problem in $(ls "$source_dir")
do
echo "-------- Starting problem: $problem"
cd "$sourcedir"
pwd
find . -name "*.c" | while read cfile; do rm "a.out" &> /dev/null; gcc "$cfile" -lm -w -fprofile-arcs -ftest-coverage -o "a.out" &> /dev/null;
echo "----- Starting program: $cfile"
if [ ! -f "a.out" ];
then
    echo "-- ERROR: $cfile did-not-compile"
else
    echo "-- COMPILED: $cfile"
    curr_dir=$(pwd)
    echo "current directory: $curr_dir";
    find "$tests_dir" -name "IN_*.txt" | while read testin; do
    # echo "--- running $testin on $cfile";
    # rm "$captured_test_output" &> /dev/null;

    # closing below statement within parantheses opens a new sub-shell and ulimit is applied only to that sub-shell.
    (ulimit -t 3; ./a.out < $testin >> /dev/null;)

    cfile_name=${cfile/"./"/""}
    cfile_name=${cfile_name/".c"/""}
    gcov "$cfile"
    mv "$cfile.gcov" "$testin-$cfile_name.gcov"
    rm "$cfile_name.gcda"

done;
fi
done;
done;


