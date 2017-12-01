CXX = c++
CXXFLAGS = -pthread -std=c++0x -fPIC
LDFLAGS = -lm
OBJ = args.o dictionary.o productquantizer.o matrix.o qmatrix.o vector.o model.o utils.o fasttext.o
OBJDIR = obj/
SRCDIR = ../src/
INCLUDES = -Iinclude/ -I../src/
SLIB = lib/libfasttext.so
ALIB = lib/libfasttext.a
AR = ar
ARFLAGS = rcs

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard ../src/*.h) Makefile

opt: CXXFLAGS += -O3 -funroll-loops
opt: obj lib $(SLIB) $(ALIB) binding

debug: CXXFLAGS += -g -O0 -fno-inline
debug: obj lib $(SLIB) $(ALIB)


$(SLIB): $(OBJS) $(OBJDIR)libfasttext.o
	$(CXX) $(INCLUDES) $(CXXFLAGS) -shared $^ -o $@ $(LDFLAGS)

$(ALIB): $(OBJS) $(OBJDIR)libfasttext.o
	$(AR) $(ARFLAGS) $(ALIB) $(OBJS)

$(OBJDIR)%.o: $(SRCDIR)%.cc $(DEPS)
	$(CXX) $(INCLUDES) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)libfasttext.o: src/libfasttext.cc include/libfasttext.h
	$(CXX) $(INCLUDES) $(CXXFLAGS) -c $< -o $@

binding:
	cd ext && phpize && ./configure && make

install:
	cd ext && make install

obj:
	mkdir -p obj

lib:
	mkdir -p lib

clean:
	rm -rf obj lib
	rm -rf ext/Makefile ext/Makefile.* ext/acinclude.m4 ext/autom4te.cache \
	       ext/aclocal.m4 ext/alibfasttextutom4te.cache ext/build \
		   ext/config.guess ext/config.h* ext/config.log ext/include \
		   ext/config.nice ext/config.status ext/config.sub ext/.libs ext/.deps \
		   ext/configure ext/configure.in ext/install-sh ext/fasttext.la \
		   ext/libtool ext/ltmain.sh ext/missing ext/fasttext.lo \
		   ext/mkinstalldirs ext/modules ext/run-tests.php
