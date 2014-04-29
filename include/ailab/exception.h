#ifndef EXCEPTION_H
#define EXCEPTION_H

namespace ailab {

class Exception {
 public:
  class IOReachedEnd {
  };
  class BadBinaryData{
  };
  class NoSuchFile {
  };
  class UnknownFileType {
  };
  class StreamIsInputOnly {
  };
  class StreamIsOutputOnly {
  };
};

}
#endif // EXCEPTION_H
