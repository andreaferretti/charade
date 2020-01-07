import { createMuiTheme } from '@material-ui/core/styles'


const theme = createMuiTheme({
  palette: {
    common: {
      black: '#000',
      white: '#fff'
    },
    background: {
      paper: '#fff',
      default: '#fff'
    },
    secondary: {
      light: 'rgba(226, 0, 26, 1)',
      main: 'rgba(226, 0, 26, 1)',
      dark: 'rgba(170, 28, 13, 1)',
      contrastText: '#fff'
    },
    primary: {
      light: 'rgba(0, 175, 208, 1)',
      main: 'rgba(0, 175, 208, 1)',
      dark: 'rgba(0, 161, 151, 1)',
      contrastText: '#fff'
    },
    error: {
      light: '#e57373',
      main: '#f44336',
      dark: '#d32f2f',
      contrastText: '#fff'
    },
    text: {
      primary: 'rgba(0, 0, 0, 0.87)',
      secondary: 'rgba(0, 0, 0, 0.54)',
      disabled: 'rgba(0, 0, 0, 0.38)',
      hint: 'rgba(0, 0, 0, 0.38)'
    }
  }
})

export default theme