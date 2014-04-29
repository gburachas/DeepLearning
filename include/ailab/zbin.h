#ifndef ZBIN_H
#define ZBIN_H

#include <stdint.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <string.h>
#include <string>
#include <limits>
#include <assert.h>

/*
 *
 *   ZBIN is a binary file format
 used in the DeepLearning Toolkit
 it benefits from being able to handle
 either sparse or dense data in the same format
 and allows for easy concatentation of data files
 e.g. cat a.zbin b.zbin > c.zbin
 yields a valid file.
 The number of columns is set to 2^16-1, if more
 are needed just change this header.
 while the number of rows is unlimited.
 *
 */
namespace zbin {

typedef uint16_t width_t;
typedef float value_t;

static const width_t denseWidth = std::numeric_limits<width_t>::max();

typedef struct {
  width_t maxWidth;
  width_t realWidth;
} RowHeader;

typedef struct {
  width_t index;
  value_t value;
} Record;

class SelectBox {

 protected:
  void _parseSection(std::string& d, size_t * start, size_t * end) {

      size_t sep = d.find(':');
      if (sep == std::string::npos) {
        *start = atol(d.c_str());
        *end = *start + 1;
      } else {

        if (sep > 0) {
          *start = atol(d.substr(0, sep).c_str());
        }

        if ((sep + 1) < d.length()) {
          *end = atol(d.substr(sep + 1, d.length()).c_str());
        }

      }

    }


 public:

  SelectBox(){
    this->startRow=0;
    this->endRow = std::numeric_limits<size_t>::max();
    this->startCol=0;
    this->endCol = std::numeric_limits<size_t>::max();
  }

  void parse(std::string& slice) {

    if(slice.length() > 0) {
      size_t comma = slice.find(',');
      if (comma == std::string::npos) {
        this->_parseSection(slice, &this->startCol, &this->endCol);
      } else {

        if (comma > 0) {
          std::string colSlice = slice.substr(0, comma);
          this->_parseSection(colSlice, &this->startCol, &this->endCol);
        }

        if ((comma + 1) < slice.length()) {
          std::string rowSlice = slice.substr(comma + 1, slice.length());
          this->_parseSection(rowSlice, &this->startRow, &this->endRow);
        }
      }
    }
  }


  size_t startRow;
  size_t endRow;
  size_t startCol;
  size_t endCol;

};


class Reader {
 protected:
  std::ifstream f;
  RowHeader header;
  size_t byteSize;
 public:
  Reader() {
    this->byteSize = 0;
    this->header.maxWidth = 0;
    this->header.realWidth = 0;
  }

  width_t width() {
    return this->header.maxWidth;
  }

  bool open(char * s) {
    this->f.open(s, std::ifstream::in | std::ifstream::binary);
    if (this->f.is_open()) {
      this->f.read((char*) &this->header, sizeof(this->header));
      this->f.seekg(0);
      this->byteSize = sizeof(value_t) * this->header.maxWidth;
      return this->f.good();
    } else {
      return false;
    }
  }

  bool next(value_t * row) {
    RowHeader rheader;
    this->f.read((char*) &rheader, sizeof(rheader));
    if (this->f.good()) {

      if (rheader.realWidth == zbin::denseWidth) {
        this->f.read((char*) row, this->byteSize);
      } else {
        Record rec;
        memset(row, 0, this->byteSize);
        for (width_t r = 0; r < rheader.realWidth; r++) {
          this->f.read((char*) &rec, sizeof(rec));
          row[rec.index] = rec.value;
        }
      }

      return true;
    } else {
      return false;
    }

  }

  void reset() {
    this->f.clear();
    this->f.seekg(0);
    assert(this->f.good());
  }

};

class Writer {
 protected:
  std::ofstream f;
  RowHeader header;

 public:
  Writer() {
    header.maxWidth = 0;
    header.realWidth = 0;
  }

  width_t width() {
    return this->header.maxWidth;
  }

  void width(width_t w) {
    this->header.maxWidth = w;
  }

  bool open(const char * s, width_t withWidth = 0) {
    this->header.maxWidth = withWidth;
    this->f.open(s, std::ofstream::out | std::ofstream::binary);
    return this->f.is_open() && this->f.good();
  }

  void write(value_t * row) {
    size_t nonzeros = 0;
    RowHeader h;
    Record r;
    h.maxWidth = this->header.maxWidth;

    for (size_t i = 0; i < this->header.maxWidth; i++) {
      nonzeros += (row[i] != 0);
    }

    if ((sizeof(value_t) * this->header.maxWidth)
        > (sizeof(Record) * nonzeros)) {

      h.realWidth = nonzeros;
      this->f.write((char*) &h, sizeof(h));
      for (size_t i = 0; i < this->header.maxWidth; i++) {
        if (row[i] != 0) {
          r.index = i;
          r.value = row[i];
          this->f.write((char*) &r, sizeof(r));
        }
      }

    } else {
      h.realWidth = zbin::denseWidth;
      this->f.write((char*) &h, sizeof(h));
      this->f.write((char*) row, sizeof(value_t) * h.maxWidth);
    }
  }

};

}
#endif
