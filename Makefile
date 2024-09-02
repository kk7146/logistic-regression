
DUMM	= data.csv
RESULT	= regression_coefficients.csv

all: $(NAME)

$(DUMM) : 
	python3 dummy.py

$(NAME) : $(DUMM)
	python3 main.py

init :
	python3 -m venv myenv
	source myenv/bin/activate
	pip install pandas scikit-learn

clean:
	rm -f $(RESULT)

fclean: clean
	rm -f $(DUMM)

re: fclean all