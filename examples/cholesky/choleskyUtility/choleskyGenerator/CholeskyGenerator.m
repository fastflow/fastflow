function MP = CholeskyGenerator(matrixSize, fileName, fileNameTransposed, fileNameCHOL)
	printf(" Generating an hermitian matrix\n");
	fflush(stdout);
	c_matrH = Hermitian(matrixSize);
	printf(" Transform the hermitian matrix in a positive definite one \n");
	fflush(stdout);
	c_matrHP = c_matrH * c_matrH';
	c_matrHP
	printf("Writing on file the Hermitian Positive Definite Matrix \n");	
	fflush(stdout);
	fp = fopen(fileName, 'w');
	fprintf(fp, '%d ', matrixSize)
	for k=1:matrixSize
		for l=1:matrixSize
			fprintf(fp, '%f ',real(c_matrHP(l,k)))
			fprintf(fp, '%f ',imag(c_matrHP(l,k)))
		end
	end
	fclose(fp)

	printf("Write the Positive Hermitian Matrix in transposed form on file \n");		
	fflush(stdout);
	fp = fopen(fileNameTransposed, 'w')
	fprintf(fp, '%d ', matrixSize)
	for k=1:matrixSize
		for l=1:matrixSize
			fprintf(fp, '%f ',real(c_matrHP(k,l)))
			fprintf(fp, '%f ',imag(c_matrHP(k,l)))
		end
	end
	fclose(fp)

	printf("Execute the Cholesky Factorization on the matrix \n");	
	fflush(stdout);
	c_matrCHOL = single(chol(single((c_matrHP))));
	printf("Write Cholesky result on file \n");		
	fflush(stdout);
	fp = fopen(fileNameCHOL, 'w');
	fprintf(fp, '%s',"MATLAB: Results of the factorization \n")
	for k=1:matrixSize
		fprintf(fp, '%s',"[ ")
		for l=1:matrixSize
			fprintf(fp, '% 6.3f ',real(c_matrCHOL(l,k)))
			fprintf(fp, '% 6.3fi ',imag(c_matrCHOL(l,k)))
		end
		fprintf(fp, '%s'," ]\n")
	end
	fprintf(fp, '%s',"\n")
	fclose(fp)
end
