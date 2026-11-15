


\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{IPCAThesisJF}[2026-08-16]
\def\extbase@baseclass{report}

%\DeclareOption{myoption}{\typeout{This is a nice option}}

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}
\ProcessOptions\relax
%\LoadClass[a4paper]{report}
%\documentclass[a4paper]{report}

\LoadClass{\extbase@baseclass}
