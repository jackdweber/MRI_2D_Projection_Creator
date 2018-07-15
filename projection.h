#ifndef h_projection
#define h_projection

#include <string>
#include <fstream>
#include <iostream>
#include <string>
#include "ImageWriter.h"

//A class that makes it easier to read in the image data.
class Projection{
public:
    int nRows = -1;
    int nCols = -1;
    int nSheets = -1;
    std::string filename;
    int pt = -1;
    std::string output;

    char* stream;


    //Reading of the file
    void readFile(){
        stream = new char[nRows*nCols*nSheets];
        std::ifstream input (filename, std::ifstream::binary);
        if(input){
            std::cout << "Reading File...\n";
            input.read(stream, (nRows*nCols*nSheets));
            std::cout << "Files read.\n";
            
        } else {
            std::cout << "File not found.\n";
        }
    }


    size_t size(){
        return nRows*nCols*nSheets* sizeof(char);
    }

    size_t imageSize(){
        if(pt == 1 || pt == 2){
            return nCols * nRows;
        }
        else if(pt == 3 || pt == 4){
            return nSheets * nRows;
        }
        else if(pt == 5 || pt == 6){
            return nSheets * nCols;
        }
        else{
            std::cout << "Error in pt type\n";
            return 0;
        }
    }




    //Function to create the image from the website
    // The input imageBytes array is xres*yres bytes. This routine will create a
    // grayscale rgb image from this array.

    void writeTheFile(std::string fName, int xres, int yres, const unsigned char* imageBytes)
    {
        unsigned char* row = new unsigned char[3*xres];
        ImageWriter* w = ImageWriter::create(fName,xres,yres);
        int next = 0;
        for (int r=0 ; r<yres ; r++)
        {
            for (int c=0 ; c<3*xres ; c+=3)
            {
                row[c] = row[c+1] = row[c+2] = imageBytes[next++];
            }
            w->addScanLine(row);
        }
        w->closeImageFile();
        delete w;
        delete [] row;
    }

};

#endif