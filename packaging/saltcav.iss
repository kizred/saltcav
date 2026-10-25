; Установщик saltcav для Windows (Inno Setup 6).
; Сборка:  iscc /DAppVersion=1.0.1 packaging\saltcav.iss   (после pyinstaller packaging\saltcav.spec)
; Результат: packaging\Output\saltcav-<версия>-setup.exe

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#define AppName "saltcav"
#define Publisher "Sergei Kizred"
#define Src "..\dist\saltcav"

[Setup]
AppId={{6B0E5C1E-4F3A-4C7B-9C55-5A1C2A7D1E01}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#Publisher}
AppCopyright=© {#Publisher}
VersionInfoVersion={#AppVersion}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
; без прав администратора — в профиль пользователя; с правами — в Program Files
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
SetupIconFile=..\src\saltcav\gui\icons\saltcav.ico
UninstallDisplayIcon={app}\saltcav-gui.exe
WizardStyle=modern
Compression=lzma2/max
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputBaseFilename={#AppName}-{#AppVersion}-setup
ChangesEnvironment=yes

[Languages]
Name: "ru"; MessagesFile: "compiler:Languages\Russian.isl"
Name: "en"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Ярлык на рабочем столе"; GroupDescription: "Ярлыки:"
Name: "addpath"; Description: "Команда saltcav в командной строке (добавить в PATH)"; GroupDescription: "Дополнительно:"; Flags: unchecked

[Files]
Source: "{#Src}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\saltcav"; Filename: "{app}\saltcav-gui.exe"; Comment: "Моделирование соляных камер"
Name: "{group}\Руководство saltcav"; Filename: "{app}\saltcav-gui.exe"; Parameters: "--help-guide"
Name: "{group}\Удалить saltcav"; Filename: "{uninstallexe}"
Name: "{autodesktop}\saltcav"; Filename: "{app}\saltcav-gui.exe"; Tasks: desktopicon

[Registry]
Root: HKCU; Subkey: "Environment"; ValueType: expandsz; ValueName: "Path"; ValueData: "{olddata};{app}"; Tasks: addpath; Check: NeedsAddPath(ExpandConstant('{app}'))

[Run]
Filename: "{app}\saltcav-gui.exe"; Description: "Запустить saltcav"; Flags: nowait postinstall skipifsilent

[Code]
function NeedsAddPath(Dir: string): Boolean;
var
  Paths: string;
begin
  if not RegQueryStringValue(HKEY_CURRENT_USER, 'Environment', 'Path', Paths) then
  begin
    Result := True;
    exit;
  end;
  Result := Pos(';' + Uppercase(Dir) + ';', ';' + Uppercase(Paths) + ';') = 0;
end;
