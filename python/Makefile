default:
	make check
unit:
	py -m unittest discover -p "*_test.py"
doctest:
	py -m doctest *.py && echo "OK"
check:
	make unit doctest
generate:
	mkdir -p inputs && py generate.py inputs
