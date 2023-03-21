#include <iostream>
#include <vector>

float randNumber(int a, int b)
{
    return rand() % (b - a + 1) + a;
}

class Matrix
{
public:
    std::vector<std::vector<float>> matrix;
    int rows, columns;
    std::string shape;
    Matrix(int a, int b)
    {
        rows = a;
        columns = b;
        for (int i = 0; i < a; i++)
        {
            std::vector<float> temp;
            for (int j = 0; j < b; j++)
            {
                temp.push_back(0);
            }
            matrix.push_back(temp);
        }
        shape = "Shape(" + std::to_string(rows) + "," + std::to_string(columns) + ")";
    }

    Matrix operator+(Matrix other)
    {
        Matrix temp(rows, columns);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < columns; j++)
                temp.matrix[i][j] = matrix[i][j] + other.matrix[i][j];
        return temp;
    }

    Matrix operator+(float a)
    {
        Matrix temp(rows, columns);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < columns; j++)
                temp.matrix[i][j] = matrix[i][j] + a;
        return temp;
    }

    Matrix operator-(Matrix other)
    {
        Matrix temp(rows, columns);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < columns; j++)
                temp.matrix[i][j] = matrix[i][j] - other.matrix[i][j];
        return temp;
    }

    Matrix operator-(float a)
    {
        Matrix temp(rows, columns);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < columns; j++)
                temp.matrix[i][j] = matrix[i][j] - a;
        return temp;
    }

    Matrix operator*(Matrix other)
    {
        Matrix temp(rows, columns);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < columns; j++)
                temp.matrix[i][j] = matrix[i][j] * other.matrix[i][j];
        return temp;
    }

    Matrix operator*(float a)
    {
        Matrix temp(rows, columns);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < columns; j++)
                temp.matrix[i][j] = matrix[i][j] * a;
        return temp;
    }

    Matrix operator/(Matrix other)
    {
        Matrix temp(rows, columns);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < columns; j++)
                temp.matrix[i][j] = matrix[i][j] / other.matrix[i][j];
        return temp;
    }

    Matrix operator/(float a)
    {
        Matrix temp(rows, columns);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < columns; j++)
                temp.matrix[i][j] = matrix[i][j] / a;
        return temp;
    }

    Matrix dot(Matrix other)
    {
        Matrix temp(rows, other.columns);
        for (int i = 0; i < rows; i++)
        {
            for (int k = 0; k < other.columns; k++)
            {
                float element = 0;
                for (int j = 0; j < columns; j++)
                {
                    element += matrix[i][j] * other.matrix[j][k];
                }
                temp.matrix[i][k] = element;
            }
        }
        return temp;
    }

    Matrix T(){
        Matrix temp(columns, rows);
        for(int i = 0; i < rows; i++)
            for(int j = 0; j < columns; j++)
                temp.matrix[j][i] = matrix[i][j];
        return temp;
    }

    float sum() {
        float temp = 0;
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < columns; j++)
                temp += matrix[i][j];
        return temp;
    }

    float mean() {
        return sum() / (rows + columns);
    }

    void randfill(int s, int e)
    {
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < columns; j++)
                matrix[i][j] = randNumber(s, e);
    }

    void print(){
        for (int i = 0; i < rows; i++){
            for (int j = 0; j < columns; j++) {
                std::cout<<matrix[i][j]<<' ';
            }
            std::cout<<'\n';
        }
        std::cout<<"Shape("<<rows<<","<<columns<<")\n";
    }
};