import "react";

declare module "react" {
  interface InputHTMLAttributes {
    webkitdirectory?: string;
    directory?: string;
  }
}
